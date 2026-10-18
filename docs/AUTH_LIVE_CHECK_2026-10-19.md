# Live auth receipt — admin endpoints — 2026-10-19

What ran: `python verify_admin.py` (the reproducible version of this
flow; it boots an isolated service on a free port with temp
databases, so nothing below touches a real `data/` directory).

Environment: v1.2.0 service on the local machine (`health` 200 before
and after the flow). Result: **14 / 14 PASS**, exit code 0.

## Checks and receipts

| # | Check | Expected | Got |
|---|-------|----------|-----|
| 1 | `POST /auth/register` (first account) | 201, role `admin` | 201, `role=admin` |
| 2 | `POST /auth/register` (second account) | 201, role `user` | 201, `role=user` |
| 3 | `GET /auth/users` (admin bearer) | 200, both accounts | 200, `[principal, intern]` |
| 4 | `GET /auth/me` (fresh user token) | 200 | 200 |
| 5 | `POST /auth/users/{name}/password` (admin) | 200, ≥1 session revoked | 200, `sessions_revoked=1` |
| 6 | old user token after the reset | 401 | 401 |
| 7 | login with the new password | 200 | 200 |
| 8 | `DELETE /auth/users/{name}` as plain user | 403 | 403 (`admin only`) |
| 9 | `DELETE /auth/users/{name}` as admin | 200 | 200 (`deleted`) |
| 10 | delete the same account again | 404 | 404 |
| 11 | admin deletes their own account | 422 | 422 (`admins cannot delete their own account`) |
| 12 | delete the only admin (store-level: HTTP self-deletion intercepts first) | `LastAdminError` | `cannot delete the only admin` |
| 13 | `POST /auth/users/{name}/revoke` | 200, sessions revoked | 200, `sessions_revoked≥1` |
| 14 | revoked token used again | 401 | 401 |

## Notes

- Check 12 is performed against the isolated `auth.db` through
  `market_intel.auth.delete_user` because over HTTP the self-deletion
  guard always answers first when the caller *is* the last admin.
- The whole transcript is reproducible: rerun `python verify_admin.py`
  any time; it never touches real data and cleans up its throwaway
  instance.
