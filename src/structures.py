# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator


class SegmentTree:
    """ Create SegmentTree.
    
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    
    Parameters
    ----------
    capacity : int
        Capacity of the tree.
    operation : function
        Function to apply to two values in the tree.
    init_value : float
        Initial value of the tree.
    """

    def __init__(self, capacity, operation, init_value):
        """Initialization.
        
        Parameters
        ----------
        capacity : int
            Capacity of the tree.
        operation : function
            Function to apply to two values in the tree.
        init_value : float
            Initial value of the tree.
        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start, end, node, node_start, node_end):
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(
                        start, mid, 2 * node, node_start, mid),
                    self._operate_helper(
                        mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start=0, end=0):
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, val):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(
                self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity):
        """Initialization.
        
        Parameters
        ----------
        capacity : int
            Capacity of the tree.
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start=0, end=0):
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound):
        """Find the highest index `i` about upper bound in the tree"""
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity):
        """Initialization.
        
        Parameters
        ----------
        capacity : int
            Capacity of the tree.
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start=0, end=0):
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)
