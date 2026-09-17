"""Runtime compatibility shims for UFL's derivative rulesets.

The Reissner-Mindlin shell forms differentiate expressions whose spatial gradient
touches a *geometric quantity* -- the bending term carries
``grad(cross(CellNormal, theta))``, so after geometry lowering both the tangent
(``derivative(F, w)``) and the shape derivative (``derivative(F, x)``) hit
``ReferenceGrad(CellNormal | Jacobian)``. Two UFL ruleset handlers need to cope
with that:

1. ``GateauxDerivativeRuleset.reference_grad`` (coefficient / state derivative) --
   a ``ReferenceGrad`` of pure geometry does not depend on the coefficient, so it
   differentiates to zero. UFL 2026.1 returns ``Zero`` when the operand carries no
   coefficients; UFL 2024.2 unconditionally raises ``NotImplementedError``.
2. ``CoordinateDerivativeRuleset.reference_grad`` (mesh-coordinate / shape
   derivative) -- has no ``Jacobian`` case and indexes ``o.ufl_operands[0]`` on the
   0-operand ``Jacobian`` terminal (IndexError / ValueError). We add the missing
   case, mirroring UFL's own dedicated ``Jacobian`` handler:

       d(RefGrad^n(J))/dX = RefGrad^n( d(J)/dX ) = RefGrad^(n+1)(v_ref)

Both rulesets were reorganised between the two releases Hermit targets:

* **UFL >= 2025** (DOLFINx 0.11) -- ``process`` is a ``singledispatchmethod``; the
  ``GateauxDerivativeRuleset`` case is already correct, and we re-``register`` the
  ``CoordinateDerivativeRuleset`` ``ReferenceGrad`` handler.
* **UFL <= 2024** (DOLFINx 0.9) -- the rulesets are plain ``MultiFunction``\\ s; we
  override the bound ``reference_grad`` methods and drop the per-class handler cache.

Remove this module once the fixes are upstream (tracked: fenics/ufl).
"""

_APPLIED = False
_MODE = None  # "singledispatch" | "multifunction" | None


def _is_singledispatch_ruleset(cls) -> bool:
    proc = cls.__dict__.get("process")
    return proc is not None and hasattr(proc, "register")


def _drop_multifunction_cache(cls) -> None:
    try:
        from ufl.corealg.multifunction import MultiFunction

        cache = getattr(MultiFunction, "_handlers_cache", None)
        if isinstance(cache, dict):
            cache.pop(cls, None)
    except Exception:
        pass


def _make_gateaux_reference_grad():
    """``ReferenceGrad`` handler for the coefficient-derivative ruleset (UFL 2024 shape)."""
    from ufl.algorithms.analysis import extract_coefficients
    from ufl.classes import Zero

    def reference_grad(self, o):
        if len(extract_coefficients(o)) > 0:
            raise NotImplementedError(
                "Currently no support for ReferenceGrad in CoefficientDerivative."
            )
        return Zero(o.ufl_shape)

    return reference_grad


def _make_coordinate_reference_grad():
    """``ReferenceGrad`` handler for the coordinate-derivative ruleset, with the
    missing ``Jacobian`` case added."""
    from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
    from ufl.classes import (
        Conj,
        FormArgument,
        Jacobian,
        ReferenceGrad,
        ReferenceValue,
        SpatialCoordinate,
    )
    from ufl.core.expr import ufl_err_str

    try:
        from ufl.domain import extract_unique_domain
    except ImportError:  # older UFL kept it as a method
        def extract_unique_domain(x):
            return x.ufl_domain()

    def reference_grad(self, g):
        o = g
        ngrads = 0
        while isinstance(o, ReferenceGrad):
            (o,) = o.ufl_operands
            ngrads += 1

        def apply_grads(f):
            for _ in range(ngrads):
                f = ReferenceGrad(f)
            return f

        # --- added case: ReferenceGrad(Jacobian) ---
        if isinstance(o, Jacobian):
            for w, v in zip(self._w, self._v):
                if extract_unique_domain(o) == extract_unique_domain(w) and isinstance(
                    remove_complex_nodes(v).ufl_operands[0], FormArgument
                ):
                    if isinstance(v, Conj):
                        return apply_grads(Conj(ReferenceGrad(remove_complex_nodes(v))))
                    return apply_grads(ReferenceGrad(v))
            return self.independent_terminal(g)

        # --- original behaviour (guarded against zero-operand terminals) ---
        _inner = o.ufl_operands[0] if o.ufl_operands else o
        if not (isinstance(o, SpatialCoordinate) or isinstance(_inner, FormArgument)):
            raise ValueError(f"Expecting gradient of a FormArgument, not {ufl_err_str(o)}")

        for w, v in zip(self._w, self._v):
            if (
                o == w
                and isinstance(v, ReferenceValue)
                and isinstance(v.ufl_operands[0], FormArgument)
            ):
                return apply_grads(v)
        return self.independent_terminal(o)

    return reference_grad


def apply() -> bool:
    """Install the UFL patches (idempotent). Returns True once installed."""
    global _APPLIED, _MODE
    if _APPLIED:
        return True

    from ufl.algorithms.apply_derivatives import (
        CoordinateDerivativeRuleset,
        GateauxDerivativeRuleset,
    )
    from ufl.classes import ReferenceGrad

    if _is_singledispatch_ruleset(CoordinateDerivativeRuleset):
        # UFL >= 2025: GateauxDerivativeRuleset already returns Zero for pure
        # geometry; only the coordinate ruleset needs the Jacobian case.
        proc = CoordinateDerivativeRuleset.__dict__["process"]
        proc.register(ReferenceGrad)(_make_coordinate_reference_grad())
        dispatcher = getattr(proc, "dispatcher", None)
        if dispatcher is not None and hasattr(dispatcher, "_clear_cache"):
            dispatcher._clear_cache()
        _MODE = "singledispatch"
    else:
        # UFL <= 2024: plain MultiFunctions -- patch both reference_grad methods.
        GateauxDerivativeRuleset.reference_grad = _make_gateaux_reference_grad()
        CoordinateDerivativeRuleset.reference_grad = _make_coordinate_reference_grad()
        _drop_multifunction_cache(GateauxDerivativeRuleset)
        _drop_multifunction_cache(CoordinateDerivativeRuleset)
        _MODE = "multifunction"

    _APPLIED = True
    return True


def installed_mode():
    """``"singledispatch"`` / ``"multifunction"`` once :func:`apply` has run, else ``None``."""
    return _MODE


def active_reference_grad_handler():
    """The coordinate-ruleset ``ReferenceGrad`` handler UFL will use (tests / debugging)."""
    from ufl.algorithms.apply_derivatives import CoordinateDerivativeRuleset
    from ufl.classes import ReferenceGrad

    if _is_singledispatch_ruleset(CoordinateDerivativeRuleset):
        return CoordinateDerivativeRuleset.__dict__["process"].dispatcher.dispatch(ReferenceGrad)
    return CoordinateDerivativeRuleset.reference_grad
