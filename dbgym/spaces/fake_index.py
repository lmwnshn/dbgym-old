from gym import spaces


class FakeIndexSpace(spaces.Discrete):
    # Historically, most people use an index space that looks like this.
    # You need to hardcode every single index that it will consider, and it is 0/1 whether the
    # index is used or not.
    pass
