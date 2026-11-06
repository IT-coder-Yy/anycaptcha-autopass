"""验证证据与预算，不把 HTTP 200、旧结果或弹窗消失当成功。"""
from dataclasses import dataclass
from enum import StrEnum
import time


class Verdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class Evidence:
    attempt_id: str
    challenge_id: str
    challenge_visible: bool
    server_accepted: bool | None = None
    response_attempt_id: str | None = None
    ui_success: bool = False
    ui_failure: bool = False
    requires_search: bool = False
    keyword: str = ""
    response_keyword: str | None = None
    search_started_after_attempt: bool = False
    search_accepted: bool = False
    # 嵌入式可保留已完成的图片；必须由当前题面的终态证明已结束。
    challenge_completed: bool = False

    def verdict(self) -> Verdict:
        current = self.response_attempt_id == self.attempt_id
        if current and self.server_accepted is False:
            return Verdict.FAILED
        if self.ui_failure:
            return Verdict.FAILED
        if not (current and self.server_accepted is True and self.ui_success
                and (not self.challenge_visible or self.challenge_completed)):
            return Verdict.UNKNOWN
        if self.requires_search and not (self.search_accepted and self.search_started_after_attempt
                                         and self.response_keyword == self.keyword):
            return Verdict.UNKNOWN
        return Verdict.PASSED


class Budget:
    def __init__(self, rounds=1, seconds=120, clock=time.monotonic):
        if rounds < 1 or seconds <= 0:
            raise ValueError("次数和时间预算必须为正数")
        self.rounds, self.seconds, self.clock = rounds, seconds, clock
        self.started, self.used = clock(), 0

    @property
    def remaining(self):
        return max(0, self.seconds - (self.clock() - self.started))

    def take(self):
        if self.used >= self.rounds or self.remaining <= 0:
            return False
        self.used += 1
        return True
