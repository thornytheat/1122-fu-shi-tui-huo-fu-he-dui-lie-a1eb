"""仅供本机无 pytest 的沙箱运行测试用的最小垫片；正式环境使用 requirements.txt 中的 pytest。"""
import math
import traceback


class Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.rel * max(abs(other), abs(self.expected)), self.abs)

    def __repr__(self):
        return f"approx({self.expected!r})"


class Mark:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        if self.name == "parametrize" and len(args) >= 2:
            argnames, argvalues = args[0], args[1]

            def decorator(func):
                names = [part.strip() for part in argnames.split(",")]

                def expanded(values):
                    params = dict(zip(names, values)) if isinstance(values, tuple) else {names[0]: values}

                    def wrapper():
                        return func(**params)

                    wrapper.__name__ = f"{func.__name__}[{values}]"
                    return wrapper

                func.__parametrized__ = [expanded(values) for values in argvalues]
                return func

            return decorator

        def decorator(func):
            return func

        return decorator


class MarkProxy:
    def __getattr__(self, name):
        return Mark(name)


class Raises:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            raise AssertionError(f"预期抛出 {self.expected.__name__}，但未抛出")
        if not issubclass(exc_type, self.expected):
            traceback.print_exception(exc_type, exc, tb)
            raise AssertionError(f"预期 {self.expected.__name__}，实际 {exc_type.__name__}")
        if self.match is not None and self.match not in str(exc):
            raise AssertionError(f"异常信息 {exc!s} 不匹配 {self.match!r}")
        self.value = exc
        return True


def approx(expected, rel=1e-6, abs=1e-12):
    return Approx(expected, rel, abs)


def raises(expected, match=None):
    return Raises(expected, match)


mark = MarkProxy()
