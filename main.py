from __future__ import annotations

import os
import json

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import sympy as sp
import mpmath as mp

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy.integrals.manualintegrate import manualintegrate

import firebase_admin
from firebase_admin import credentials, auth


# =========================
# Firebase Admin (Render Secret)
# =========================
if not firebase_admin._apps:
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing FIREBASE_SERVICE_ACCOUNT_JSON env var")

    cred_dict = json.loads(raw)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)


def verify_token(authorization: str | None):
    """
    ✅ لازم Authorization: Bearer <Firebase ID Token>
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.split("Bearer ")[1].strip()
    try:
        decoded = auth.verify_id_token(token)
        return decoded  # فيها uid
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# =========================
# APP
# =========================
app = FastAPI(title="Tadriss DZ - Smart Calculator Backend", version="1.0.0")

x = sp.Symbol("x", real=True)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


# =========================
# Request Model
# =========================
class CalcReq(BaseModel):
    tool: str                  # derivative | limit | integral | primitive
    expr: str
    point: str | None = None   # limit point: "0" "+oo" "-oo"
    side: str | None = None    # "+", "-", "both"
    a: str | None = None       # integral lower
    b: str | None = None       # integral upper


# =========================
# Parsing helpers
# =========================
def parse_point(p: str):
    p = (p or "").strip().lower()

    if p in ["+oo", "oo", "+inf", "inf", "+infty", "infty", "+infinity", "infinity"]:
        return sp.oo
    if p in ["-oo", "-inf", "-infty", "-infinity"]:
        return -sp.oo

    return parse_expr(
        p,
        local_dict={"x": x, "pi": sp.pi, "π": sp.pi, "e": sp.E, "E": sp.E},
        transformations=TRANSFORMS,
    )


def parse_math(expr: str):
    s = (expr or "").strip()

    # دعم رموز شائعة من Flutter
    s = s.replace("π", "pi")
    s = s.replace("×", "*").replace("÷", "/")

    local = {
        "x": x,
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "ln": sp.log,
        "log": sp.log,
        "abs": sp.Abs,
        "Abs": sp.Abs,
    }

    return parse_expr(s, local_dict=local, transformations=TRANSFORMS)


# =========================
# LaTeX safe output
# =========================
def _latex_text_fallback(s: str) -> str:
    s = s.replace("\\", r"\\")
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("&", r"\&")
    return r"\text{" + s + "}"


def safe_latex(sym_expr) -> str:
    """
    ✅ يضمن دائماً LaTeX صحيح حتى لو النتيجة كانت نصية أو شيء ما
    """
    try:
        latex = sp.latex(sym_expr, ln_notation=True)
        latex = latex.replace(r"\log", r"\ln")
        return latex
    except Exception:
        return _latex_text_fallback(str(sym_expr))


# =========================
# Validation helpers
# =========================
def is_bad_value(v) -> bool:
    """
    قيم غير مقبولة:
    - NaN / zoo
    - Complex
    - AccumBounds (لا توجد نهاية)
    - Limit(...) أو Integral(...) (غير محسوب)
    """
    try:
        if v is None:
            return True

        if isinstance(v, (sp.Limit, sp.Integral, sp.AccumBounds)):
            return True

        if v in [sp.nan, sp.zoo]:
            return True

        if hasattr(v, "has") and v.has(sp.I):
            return True

        s = str(v).lower()
        bad = ["nan", "zoo", "complex", "accumbounds"]
        return any(b in s for b in bad)
    except Exception:
        return True


# =========================
# "School" simplification
# =========================
def school_simplify(expr: sp.Expr) -> sp.Expr:
    """
    ✅ تبسيط مدرسي عام
    """
    try:
        e = sp.together(expr)
        e = sp.cancel(e)
        e = sp.factor(e)
        e = sp.simplify(e)
        return e
    except Exception:
        return sp.simplify(expr)


def school_derivative_form(expr: sp.Expr) -> sp.Expr:
    """
    ✅ شكل مدرسي: نجمع حدود ln في معامل واحد داخل البسط (collect)
    """
    try:
        r = sp.together(expr)
        num, den = sp.fraction(r)

        num = sp.expand(num)
        den = sp.factor(den)

        # نجمع حسب log(...) (كل لوغاريتم يظهر)
        logs = sorted(list(num.atoms(sp.log)), key=lambda t: str(t))
        for L in logs:
            num = sp.collect(num, L)

        return sp.Mul(num, sp.Pow(den, -1, evaluate=False), evaluate=False)

    except Exception:
        return school_simplify(expr)


# =========================
# LIMIT helper
# =========================
def compute_limit(expr, point_str: str, side: str):
    p = parse_point(point_str)
    side = (side or "both").strip().lower()

    try:
        if side == "+":
            L = sp.limit(expr, x, p, dir="+")
        elif side == "-":
            L = sp.limit(expr, x, p, dir="-")
        else:
            L = sp.limit(expr, x, p)

        if isinstance(L, sp.AccumBounds):
            return None, "لا توجد نهاية"

        if isinstance(L, sp.Limit):
            return None, "لا توجد نهاية"

        if is_bad_value(L):
            return None, "لا يمكن حساب هذه العملية"

        return school_simplify(L), None
    except Exception:
        return None, "لا توجد نهاية"


# =========================
# Numeric quad helper (fallback for definite integral)
# =========================
def _numeric_quad(expr_sym: sp.Expr, a, b):
    f = sp.lambdify(x, expr_sym, "mpmath")
    mp.mp.dps = 50

    def g(t):
        return f(t)

    # حدود لا نهائية
    if a == -sp.oo and b == sp.oo:
        return mp.quad(g, [-mp.inf, mp.inf])
    if a == -sp.oo:
        return mp.quad(g, [-mp.inf, float(sp.N(b))])
    if b == sp.oo:
        return mp.quad(g, [float(sp.N(a)), mp.inf])

    return mp.quad(g, [float(sp.N(a)), float(sp.N(b))])


def compute_def_integral(expr_sym: sp.Expr, a, b):
    """
    ✅ تكامل محدد:
    1) integrate محدد مباشرة
    2) F(b)-F(a) باستخدام primitive
    3) numeric quad كحل أخير
    """
    # 1) direct definite
    try:
        r1 = sp.integrate(expr_sym, (x, a, b))
        if not (
            isinstance(r1, sp.Integral)
            or (hasattr(r1, "has") and r1.has(sp.Integral))
            or is_bad_value(r1)
        ):
            return school_simplify(r1), None
    except Exception:
        pass

    # 2) antiderivative then evaluate
    F = None
    try:
        F = sp.integrate(expr_sym, x)
        if isinstance(F, sp.Integral) or (hasattr(F, "has") and F.has(sp.Integral)) or is_bad_value(F):
            F = None
    except Exception:
        F = None

    if F is None:
        try:
            F = manualintegrate(expr_sym, x)
            if isinstance(F, sp.Integral) or (hasattr(F, "has") and F.has(sp.Integral)) or is_bad_value(F):
                F = None
        except Exception:
            F = None

    if F is not None:
        try:
            r2 = sp.simplify(F.subs(x, b) - F.subs(x, a))
            if not is_bad_value(r2):
                return school_simplify(r2), None
        except Exception:
            pass

    # 3) numeric
    try:
        val = _numeric_quad(expr_sym, a, b)
        r3 = sp.nsimplify(str(val)) if mp.isfinite(val) else sp.Float(val)
        return r3, None
    except Exception:
        return None, "لا يمكن حساب هذا التكامل"


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/calc")
def calc(req: CalcReq, authorization: str | None = Header(default=None)):
    # ✅ حماية: لازم توكن
    verify_token(authorization)

    try:
        expr = parse_math(req.expr)
        tool = (req.tool or "").lower().strip()

        # -------- derivative --------
        if tool == "derivative":
            r = sp.diff(expr, x)
            r = school_derivative_form(r)

            if is_bad_value(r):
                return {"ok": False, "error": "لا يمكن حساب هذه العملية"}

            return {
                "ok": True,
                "result_latex": safe_latex(r),
                "result_text": str(r),
            }

        # -------- primitive --------
        elif tool == "primitive":
            r = sp.integrate(expr, x)

            if isinstance(r, sp.Integral) or (hasattr(r, "has") and r.has(sp.Integral)) or is_bad_value(r):
                try:
                    r = manualintegrate(expr, x)
                except Exception:
                    return {"ok": False, "error": "لا يمكن حساب الدالة الأصلية"}

            if isinstance(r, sp.Integral) or (hasattr(r, "has") and r.has(sp.Integral)) or is_bad_value(r):
                return {"ok": False, "error": "لا يمكن حساب الدالة الأصلية"}

            r = school_simplify(r)

            return {
                "ok": True,
                "result_latex": safe_latex(r) + r" + C",
                "result_text": str(r) + " + C",
            }

        # -------- integral (definite) --------
        elif tool == "integral":
            if req.a is None or req.b is None:
                return {"ok": False, "error": "حددي a و b"}

            a = parse_point(req.a)
            b = parse_point(req.b)

            r, err = compute_def_integral(expr, a, b)
            if err:
                return {"ok": False, "error": err}

            return {
                "ok": True,
                "result_latex": safe_latex(r),
                "result_text": str(r),
            }

        # -------- limit --------
        elif tool == "limit":
            if req.point is None:
                return {"ok": False, "error": "حددي نقطة النهاية"}

            L, err = compute_limit(expr, req.point, req.side or "both")
            if err:
                return {"ok": False, "error": err}

            return {
                "ok": True,
                "result_latex": safe_latex(L),
                "result_text": str(L),
            }

        else:
            return {"ok": False, "error": "Unknown tool"}

    except Exception as e:
        return {"ok": False, "error": f"{e}"}
