"""How the Digital Twin routes reject a request without echoing library text (#2020).

The app's HTTPException handler copies a 400's detail verbatim into the client's message, and
the ValueErrors these routes catch are not authored by them (pydantic, enum and UUID text). So
the client gets a fixed sentence and the raw text goes to the route's log. Kept out of
``digital_twin.py``, which the module-size ratchet pins (#1991 debt 4).
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import HTTPException
from pydantic import Field

#: A target decile, bounded in the request model and not only on the domain InterventionConfig:
#: an out-of-range decile used to pass request validation and fail inside /simulate's try, which
#: returned pydantic's text.
Decile = Annotated[int, Field(ge=1, le=10)]


def rejected_request(
    logger: logging.Logger, subject: str, parameters: str, exc: ValueError
) -> HTTPException:
    """A 400 naming the ``subject`` request and the ``parameters`` to check; ``exc``, with its
    traceback, goes to ``logger`` (the route's own, so its records stay under the route)."""
    logger.warning("%s request rejected: %s", subject.capitalize(), exc, exc_info=exc)
    return HTTPException(
        status_code=400,
        detail=(
            f"The {subject} request could not be processed. Check the {parameters} "
            "parameters and try again."
        ),
    )
