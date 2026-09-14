"""
The structure-version TIMELINE of an expert review.

``expert_review_versions`` (migration 141) records every DAG structure a run
produced for a review, oldest first -- a TIMELINE, not a set, so a revert
(A -> B -> A) appends a third row. Beside it lives the review row's own PAIR
compare-and-set (``dag_version_hash`` + ``adjustment_set_hash``, migration
142), which is what moves the review ONTO a version; recording a structure and
moving the review onto it are two decisions, and this module offers them
separately because the gate makes them separately.

Split out of ``expert_review.py`` by concern (#1991 debts 3/4): that module had
grown past the module-size ratchet, and the timeline is the half of it that
answers a different question -- what structures a review has been through,
rather than what the review itself says.
"""

import logging
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository
from src.repositories.json_utils import to_plain_json
from src.repositories.query_utils import match_nullable_column

logger = logging.getLogger(__name__)


class ExpertReviewVersionTimeline(BaseRepository):
    """The timeline half of :class:`~src.repositories.expert_review.ExpertReviewRepository`.

    Not instantiated on its own -- ``ExpertReviewRepository`` inherits it, so
    ``self.client`` and ``self.table_name`` are the review repository's and
    every caller reaches these methods as ``repo.<method>``, exactly as before
    the split.
    """

    async def record_version(
        self,
        review_id: str,
        *,
        dag_version_hash: str,
        dag_structure: Optional[Dict[str, Any]],
        adjustment_set_hash: Optional[str],
        query_id: Optional[str],
    ) -> bool:
        """Write one structure version to the review's TIMELINE. The insert only
        -- the review row is not touched.

        Recording a structure and MOVING the review onto it are two decisions,
        and the gate makes them separately, so the repository offers them
        separately. A review that already carries the pair it is recording needs
        the timeline row and nothing else: routing that case through
        :meth:`append_version` re-wrote the review with the pair it already had,
        which cleared ``agent_assessment_json`` -- discarding an advisory grading
        OF THAT VERY STRUCTURE, for no reason. Here there is no UPDATE at all, so
        nothing to clear and nothing to compare-and-set.

        ``expert_review_versions`` is a TIMELINE (migration 141), not a set: a
        revert (A -> B -> A) records a third row rather than being suppressed, so
        same-pair idempotence is the caller's gate, not a constraint's.
        ``insert()``, never ``upsert()``: ``upsert`` defaults to ON CONFLICT DO
        UPDATE and ``service_role`` holds SELECT+INSERT only on this table
        (42501 otherwise).

        Args:
            review_id: The review whose timeline gains a row
            dag_version_hash: The structure's DAG hash
            dag_structure: The sanitized snapshot, or None when there is none
            adjustment_set_hash: The adjustment-set half; None means UNKNOWN,
                which is what migration 141's backfilled rows carry
            query_id: The run that produced the structure

        Returns:
            True when the row was inserted; False on a persistence error, which
            is logged. The caller decides what a failed timeline write means for
            the review -- ``append_version`` refuses to advance after one.
        """
        if not self.client:
            return False

        try:
            await (
                self.client.table("expert_review_versions")
                .insert(
                    {
                        "review_id": review_id,
                        "dag_version_hash": dag_version_hash,
                        "dag_structure_json": to_plain_json(dag_structure)
                        if dag_structure
                        else None,
                        "adjustment_set_hash": adjustment_set_hash,
                        "query_id": query_id,
                    }
                )
                .execute()
            )
        except Exception as e:
            logger.error(
                f"record_version: insert failed for review {review_id} at "
                f"({dag_version_hash}, {adjustment_set_hash}): {e}"
            )
            return False
        return True

    async def advance_review(
        self,
        review_id: str,
        *,
        dag_version_hash: str,
        dag_structure: Optional[Dict[str, Any]],
        adjustment_set_hash: Optional[str],
        related_validation_ids: Optional[List[str]] = None,
        expected_current_hash: Optional[str],
        expected_current_adjustment_hash: Optional[str],
    ) -> bool:
        """Move a PENDING review onto a structure version, as a COMPARE-AND-SET on
        the pair it currently carries. The review UPDATE only -- no version row.

        Split out of ``append_version`` in codex round 2, because an advance and
        an append are two decisions, not one. A review can be BEHIND the
        timeline's latest row without anything new to append: two adjustment-only
        advances both insert, one loses the compare-and-set, and the loser's
        review is then on a pair the timeline already holds. The repair is an
        advance with NO append -- appending again would add a duplicate row to
        say something the timeline already says. The gate decides the two
        separately and calls whichever applies.

        The compare-and-set is on the PAIR (codex round-2 HIGH 3). Under the old
        hash-only CAS, two concurrent advances from (h1, A) to (h1, B) and
        (h1, C) BOTH matched -- the DAG hash is h1 throughout -- so the review
        ended on whichever landed last while the timeline's latest row was the
        other. A later run of that pair then found both the review's hash and the
        timeline's latest pair matching and skipped forever: a permanent strand,
        with no outage required. Comparing both halves makes exactly one of the
        two win, and the loser returns False and is logged.

        The payload carries ``adjustment_set_hash`` as a LITERAL None when the
        adjustment set is unknown, never by omitting the key. An omission would
        leave the PREVIOUS hash on a row whose structure has just been replaced
        -- a value that looks like a measurement of the new structure and is a
        measurement of the old one.

        ``dag_structure`` of None CLEARS the snapshot, deliberately: the snapshot
        is what the review UI renders, and the previous structure under a new
        identity would show a DAG the review no longer covers.
        ``related_validation_ids`` is omitted when None so a caller without ids
        never NULLs the column.

        Args:
            review_id: The review to advance
            dag_version_hash: The structure hash it moves TO
            dag_structure: The sanitized snapshot it moves to; None clears it
            adjustment_set_hash: The adjustment-set half it moves to; None is
                written as SQL NULL (unknown)
            related_validation_ids: This run's evidence, when the caller has it
            expected_current_hash: The DAG hash the caller READ before deciding
                (keyword-only and required: an advance that cannot name what it
                is replacing is not a compare-and-set). None means the row's
                hash is NOT RECORDED -- matched with IS NULL, never skipped
            expected_current_adjustment_hash: The adjustment half it read, None
                for "still unknown" -- matched with IS NULL, never skipped

        Returns:
            True when exactly this pending row at exactly that pair was
            advanced; False on a lost race, a resolved review, a nonexistent one,
            or a persistence error
        """
        if not self.client:
            return False

        advance: Dict[str, Any] = {
            "dag_version_hash": dag_version_hash,
            "dag_structure_json": to_plain_json(dag_structure) if dag_structure else None,
            # Migration 142: the row's own copy of the adjustment-set half. A
            # LITERAL None when unknown -- see the docstring on why an omission
            # would strand a stale hash on a replaced structure.
            "adjustment_set_hash": adjustment_set_hash,
            # The cached advisory assessment grades the DAG and the evidence of
            # the version that WAS current, so it cannot survive the advance
            # (codex round-1 HIGH): the review UI would show a grading of a
            # structure this review no longer covers, and the assessment route's
            # cache short-circuit would keep serving it beside the new snapshot.
            # A literal None, not an omission: this payload is built explicitly,
            # so it is never dropped by the "remove None values" pattern
            # ``submit_review`` uses, and PostgREST writes it as SQL NULL.
            "agent_assessment_json": None,
        }
        if related_validation_ids is not None:
            advance["related_validation_ids"] = related_validation_ids

        try:
            query = (
                self.client.table(self.table_name)
                .update(advance)
                .eq("review_id", review_id)
                .eq("approval_status", "pending")
            )
            # BOTH halves of the pair through the same eq-or-IS-NULL filter: a
            # pending row whose hash was never recorded is a row to compare
            # against, not one to skip -- an ``eq`` on None would match nothing
            # and strand it (an appended version row the review never reaches).
            query = match_nullable_column(query, "dag_version_hash", expected_current_hash)
            result = await match_nullable_column(
                query, "adjustment_set_hash", expected_current_adjustment_hash
            ).execute()
        except Exception as e:
            logger.error(
                f"advance_review: review {review_id} could not be advanced to "
                f"({dag_version_hash}, {adjustment_set_hash}): {e}"
            )
            return False

        if not result.data:
            logger.warning(
                f"advance_review: review {review_id} is no longer pending on "
                f"({expected_current_hash}, {expected_current_adjustment_hash}), so it was "
                f"not advanced to ({dag_version_hash}, {adjustment_set_hash}) -- a concurrent "
                "advance won, or the review was resolved; the caller should re-read."
            )
            return False
        return True

    async def append_version(
        self,
        review_id: str,
        *,
        dag_version_hash: str,
        dag_structure: Optional[Dict[str, Any]],
        adjustment_set_hash: Optional[str],
        query_id: Optional[str],
        related_validation_ids: Optional[List[str]] = None,
        expected_current_hash: Optional[str],
        expected_current_adjustment_hash: Optional[str],
    ) -> bool:
        """Record a new structure version on a PENDING review and make it the
        review's current hash.

        The timeline INSERT plus :meth:`advance_review` -- the two halves of
        "this run produced a structure nobody has recorded yet". An advance can
        also happen WITHOUT an append (a review left behind by a lost
        compare-and-set, where the timeline already holds the pair); the gate
        decides append and advance independently and calls whichever applies.

        ``expert_review_versions`` is a TIMELINE (migration 141), not a set: the
        caller appends only when the hash differs from the review's current one,
        and a revert (A -> B -> A) appends a third row rather than being
        suppressed -- so same-hash idempotence is the caller's gate, not a
        constraint's. ``insert()``, never ``upsert()``: ``upsert`` defaults to ON
        CONFLICT DO UPDATE, and ``service_role`` holds SELECT+INSERT only on this
        table (42501 otherwise).

        The review update touches ``dag_version_hash``, ``adjustment_set_hash``
        (migration 142), ``dag_structure_json`` and -- when given --
        ``related_validation_ids``. The version row keeps its OWN
        ``adjustment_set_hash``: that column is the HISTORY, the review's is its
        CURRENT state, and they answer different questions.
        The evidence ids go on the REVIEW rather than the version because
        they describe its current state: the detail route renders evidence from
        that column, and after an advance the previous run's ids would show
        statistics computed on a structure the review no longer covers. Omitted
        when None, so a caller without ids in hand never NULLs the column. A ``dag_structure`` of None CLEARS the
        review's snapshot, deliberately: the snapshot is what the review UI
        renders, and leaving the previous structure under a NEW hash would show
        a DAG the review no longer covers. (``update_dag_structure`` refuses an
        empty structure for the opposite reason -- it BACKFILLS the structure of
        the hash already on the row, so there a None would erase and replace
        nothing.)

        ``expected_current_hash`` and ``expected_current_adjustment_hash`` make
        the advance a COMPARE-AND-SET on the PAIR (codex round-1 HIGH, widened
        to both halves in round 2): of two interleaved appends (insert h2,
        insert h3, advance h3, advance h2) the second advance matches zero rows
        instead of dragging the review back to a structure a later run
        superseded -- and because both halves are compared, that holds for two
        ADJUSTMENT-ONLY advances too, where the DAG hash is identical throughout.
        The caller passes the pair it READ before deciding to append. BOTH are
        required now (the optional-CAS path is gone): every caller reads the
        review row before deciding, so a caller with no pair in hand is a caller
        that has not made the decision this method executes. Either half may be
        None -- "the row does not record this" is a precondition to match with
        IS NULL, not one to skip (see :func:`match_nullable_column`).

        On a lost race the version row it already inserted STAYS: the
        timeline is a record of the structures runs produced, and it may
        therefore hold a row the review never pointed at. Its LAST row is still
        the truth of what was last recorded, and the review's own pair is what a
        resolution binds to (``submit_review``'s ``expected_dag_version_hash`` /
        ``expected_adjustment_set_hash``) -- so an orphan version row can never
        widen what a reviewer signed off. The review left behind is repaired by
        the next run of that pair through :meth:`advance_review`, with no
        append: the strand codex round 2 found is closed by that second path,
        not by this one.

        What the compare-and-set does NOT prevent is a duplicate row. Two
        concurrent pending-branch advances to the SAME new structure can both
        read the old latest version, both decide the pair differs, and both
        insert before either advances -- the timeline then holds that structure
        twice. The insert happens before the CAS and nothing serialises it; the
        CAS stops the review REGRESSING to a superseded hash, not the second
        row. Closing that too needs DB-side serialization (an RPC), deliberately
        not done: a mint or an advance is rare, and a duplicated version row
        overstates how often the DAG changed without misreporting what it is.

        The case that used to cost something here no longer reaches this
        method: when the gate needs the TIMELINE row but not an advance -- the
        latest recorded pair is an orphan from a lost race while the review
        already carries this run's pair -- it calls :meth:`record_version`
        directly. Routing that through here re-wrote the review with the pair it
        already had and cleared ``agent_assessment_json``, discarding an advisory
        grading of that very structure. The review row is now untouched in that
        case, so there is nothing to clear and no compare-and-set to lose.

        Returns:
            True only when BOTH the append and the review's advance succeeded.
            False when the append failed (nothing was written and the review is
            untouched), when the review advance failed, or when it matched no
            PENDING row -- a resolved review is not advanced, and claiming True
            there would report a current hash the row does not carry. After a
            False that follows a successful append the timeline is one row ahead
            of the review; the caller re-reads rather than treating False as
            "nothing was written".
        """
        if not self.client:
            return False

        # The timeline write is the SAME operation the record-only path runs, so
        # it is the same code: one definition of "write this structure to the
        # timeline", one insert payload, one failure log.
        if not await self.record_version(
            review_id,
            dag_version_hash=dag_version_hash,
            dag_structure=dag_structure,
            adjustment_set_hash=adjustment_set_hash,
            query_id=query_id,
        ):
            return False

        # The advance is likewise the SAME operation the repair path runs: one
        # definition of "move the review onto this pair", one compare-and-set,
        # one warning. Its False already names both expected halves; the
        # append's own consequence -- the version row stays, one ahead of the
        # review -- is documented above.
        return await self.advance_review(
            review_id,
            dag_version_hash=dag_version_hash,
            dag_structure=dag_structure,
            adjustment_set_hash=adjustment_set_hash,
            related_validation_ids=related_validation_ids,
            expected_current_hash=expected_current_hash,
            expected_current_adjustment_hash=expected_current_adjustment_hash,
        )

    async def get_latest_version(self, review_id: str) -> Optional[Dict[str, Any]]:
        """The LAST structure version recorded for a review, or None.

        The other end of ``get_versions``' order -- newest ``created_at`` first,
        ties broken by ``version_id`` -- taken in ONE query with ``limit(1)``
        rather than by fetching a whole timeline to read its last element. The
        gate compares the run's ``(dag_version_hash, adjustment_set_hash)`` pair
        against this row to decide whether anything actually changed, which is
        what keeps two concurrent same-structure mints from appending the same
        pair twice (the 23505 recovery hands both of them the winner's review).

        Returns:
            The newest version row, or None when the review has no timeline (a
            fresh insert, or a pre-141 row the backfill skipped) or no client

        Raises:
            The underlying client error on a query failure, after logging it
            (R1/R3 convention of this module): None reads as "nothing recorded
            yet, append", so an outage must not fake it into a duplicate append.
        """
        if not self.client:
            return None

        try:
            result = await (
                self.client.table("expert_review_versions")
                .select("*")
                .eq("review_id", review_id)
                .order("created_at", desc=True)
                .order("version_id", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to get the latest version for review {review_id}: {e}")
            raise

        rows = result.data or []
        return dict(rows[0]) if rows else None

    async def get_versions(self, review_id: str) -> List[Dict[str, Any]]:
        """Structure versions of a review, OLDEST first; ties on ``created_at``
        broken by ``version_id`` so a backfill that stamped several rows with the
        same timestamp still has one stable order.

        Returns:
            The review's version rows in timeline order

        Raises:
            The underlying client error on a query failure, after logging it
            (R1/R3 convention): an empty timeline reads as "the structure never
            changed". The no-client early return ([]) is unchanged.
        """
        if not self.client:
            return []

        try:
            result = await (
                self.client.table("expert_review_versions")
                .select("*")
                .eq("review_id", review_id)
                .order("created_at", desc=False)
                .order("version_id", desc=False)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get versions for review {review_id}: {e}")
            raise

    async def get_versions_for_reviews(
        self, review_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Structure versions for MANY reviews in ONE query, grouped by review id.

        The batched sibling of ``get_versions``: the pending queue renders up to
        200 rows and needs each row's version count, which per-row calls would
        turn into 200 round trips. Each group is in the same timeline order
        ``get_versions`` returns -- OLDEST first, ties on ``created_at`` broken
        by ``version_id`` -- so a caller can read the last element as the latest
        version. The ordering is applied to the whole result set before
        grouping, which preserves it within every group.

        A review with no version rows is ABSENT from the mapping rather than
        present with an empty list: the caller decides what "no timeline" means
        for it (the queue reads it as version 1, changed at creation).

        Args:
            review_ids: The review ids to fetch versions for; an empty list is
                an empty mapping and no query

        Returns:
            ``{review_id: [version rows, oldest first]}`` for the ids that have
            versions

        Raises:
            The underlying client error on a query failure, after logging it
            (R1/R3 convention of this module): an empty mapping reads as "none
            of these reviews ever changed", which an outage must not fake. The
            no-client early return ({}) is unchanged.
        """
        if not self.client or not review_ids:
            return {}

        try:
            result = await (
                self.client.table("expert_review_versions")
                .select("*")
                .in_("review_id", review_ids)
                .order("created_at", desc=False)
                .order("version_id", desc=False)
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to get versions for {len(review_ids)} reviews: {e}")
            raise

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in result.data or []:
            review_id = row.get("review_id")
            if review_id is None:
                continue
            grouped.setdefault(str(review_id), []).append(row)
        return grouped
