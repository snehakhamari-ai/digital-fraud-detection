def risk_factors(r):
    """Return human-readable signals using the transaction's raw feature names."""
    out = []
    if r["new_device"]:
        out.append("New device")
    if r["location_mismatch"]:
        out.append("Location mismatch")
    if r["is_new_payee"]:
        out.append("First payment to this payee")
    if r["amount_ratio"] >= 3:
        out.append(f"Amount {r['amount_ratio']:.1f}x customer's usual")
    if r["hour"] in (0, 1, 2, 3):
        out.append("Night-time payment")
    if r["txns_last_hour"] >= 3:
        out.append("Many payments in last hour")
    if r["merchant_type"] in ("crypto", "gaming"):
        out.append("High-risk merchant type")
    return ", ".join(out) if out else "No major risk factor"