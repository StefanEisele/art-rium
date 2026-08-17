"""Cloud video generation (MiniMax H3) with a hard monthly spending limit.

    pricing.py   what a call costs — pure arithmetic over a documented rate card
    budget.py    the ledger: reserve → settle / release, under a row lock
    backend.py   the provider interface, and MiniMax behind it
    queue.py     submit, poll, download, settle — all state in the database

Read budget.py first. Everything else is replaceable; the two-phase booking is
the part that keeps a bug from becoming a bill.
"""
