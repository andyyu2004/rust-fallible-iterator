use super::FoldStop;
use core::cell::Cell;
use core::cmp;

#[cfg(feature = "std")]
mod imports {
    pub use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
    pub use std::hash::{BuildHasher, Hash};
    pub use std::prelude::v1::*;
}

#[cfg(any(feature = "std", feature = "alloc"))]
use crate::imports::*;
use crate::{ChainState, ResultExt};

/// An `Iterator`-like trait that allows for calculation of items to fail.
pub trait FallibleLendingIterator {
    type Item<'a>;
    type Error;

    fn next<'a>(&'a mut self) -> Result<Option<Self::Item<'a>>, Self::Error>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    /// Returns an iterator which applies a fallible transform to the elements
    /// of the underlying iterator.
    #[inline]
    fn map<F, B>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>) -> Result<B, Self::Error>,
    {
        Map { it: self, f }
    }

    /// Calls a fallible closure on each element of an iterator.
    #[inline]
    fn for_each<F>(self, mut f: F) -> Result<(), Self::Error>
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>) -> Result<(), Self::Error>,
    {
        self.fold((), move |(), item| f(item))
    }

    /// Returns an iterator which uses a predicate to determine which values
    /// should be yielded. The predicate may fail; such failures are passed to
    /// the caller.
    #[inline]
    fn filter<F>(self, f: F) -> Filter<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item<'_>) -> Result<bool, Self::Error>,
    {
        Filter { it: self, f }
    }

    /// Returns an iterator which both filters and maps. The closure may fail;
    /// such failures are passed along to the consumer.
    #[inline]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>) -> Result<Option<B>, Self::Error>,
    {
        FilterMap { it: self, f }
    }

    /// Returns an iterator which yields the current iteration count as well
    /// as the value.
    #[inline]
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
    {
        Enumerate { it: self, n: Default::default() }
    }

    /// Returns an iterator which applies a stateful map to values of this
    /// iterator.
    #[inline]
    fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
    where
        Self: Sized,
        F: FnMut(&mut St, Self::Item<'_>) -> Result<Option<B>, Self::Error>,
    {
        Scan { it: self, f, state: initial_state }
    }

    /// Returns an iterator which yields this iterator's elements and ends after
    /// the first `Ok(None)`.
    ///
    /// The behavior of calling `next` after it has previously returned
    /// `Ok(None)` is normally unspecified. The iterator returned by this method
    /// guarantees that `Ok(None)` will always be returned.
    #[inline]
    fn fuse(self) -> Fuse<Self>
    where
        Self: Sized,
    {
        Fuse { it: self, done: false }
    }

    /// Returns an iterator which passes each element to a closure before returning it.
    #[inline]
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item<'_>) -> Result<(), Self::Error>,
    {
        Inspect { it: self, f }
    }

    /// Borrow an iterator rather than consuming it.
    ///
    /// This is useful to allow the use of iterator adaptors that would
    /// otherwise consume the value.
    #[inline]
    fn by_ref(&mut self) -> &mut Self
    where
        Self: Sized,
    {
        self
    }

    /// Applies a function over the elements of the iterator, producing a single
    /// final value.
    #[inline]
    fn fold<B, F>(mut self, init: B, f: F) -> Result<B, Self::Error>
    where
        Self: Sized,
        F: FnMut(B, Self::Item<'_>) -> Result<B, Self::Error>,
    {
        self.try_fold(init, f)
    }

    /// Applies a function over the elements of the iterator, producing a single final value.
    ///
    /// This is used as the "base" of many methods on `FallibleLendingIterator`.
    #[inline]
    fn try_fold<B, E, F>(&mut self, mut init: B, mut f: F) -> Result<B, E>
    where
        Self: Sized,
        E: From<Self::Error>,
        F: FnMut(B, Self::Item<'_>) -> Result<B, E>,
    {
        while let Some(v) = self.next()? {
            init = f(init, v)?;
        }
        Ok(init)
    }

    /// Determines if all elements of this iterator match a predicate.
    #[inline]
    fn all<F>(&mut self, mut f: F) -> Result<bool, Self::Error>
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>) -> Result<bool, Self::Error>,
    {
        self.try_fold((), |(), v| {
            if !f(v)? {
                return Err(FoldStop::Break(false));
            }
            Ok(())
        })
        .map(|()| true)
        .unpack_fold()
    }

    /// Determines if any element of this iterator matches a predicate.
    #[inline]
    fn any<F>(&mut self, mut f: F) -> Result<bool, Self::Error>
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>) -> Result<bool, Self::Error>,
    {
        self.try_fold((), |(), v| {
            if f(v)? {
                return Err(FoldStop::Break(true));
            }
            Ok(())
        })
        .map(|()| false)
        .unpack_fold()
    }

    /// Returns the position of the first element of this iterator that matches
    /// a predicate. The predicate may fail; such failures are returned to the
    /// caller.
    #[inline]
    fn position<F>(&mut self, mut f: F) -> Result<Option<usize>, Self::Error>
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>) -> Result<bool, Self::Error>,
    {
        self.try_fold(0, |n, v| {
            if f(v)? {
                return Err(FoldStop::Break(Some(n)));
            }
            Ok(n + 1)
        })
        .map(|_| None)
        .unpack_fold()
    }

    /// Returns an iterator that yields this iterator's items in the opposite
    /// order.
    #[inline]
    fn rev(self) -> Rev<Self>
    where
        Self: Sized + DoubleEndedFallibleLendingIterator,
    {
        Rev(self)
    }

    /// Returns an iterator which applies a transform to the errors of the
    /// underlying iterator.
    #[inline]
    fn map_err<B, F>(self, f: F) -> MapErr<Self, F>
    where
        F: FnMut(Self::Error) -> B,
        Self: Sized,
    {
        MapErr { it: self, f }
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]

/// A fallible iterator able to yield elements from both ends.
pub trait DoubleEndedFallibleLendingIterator: FallibleLendingIterator {
    /// Advances the end of the iterator, returning the last value.
    fn next_back(&mut self) -> Result<Option<Self::Item<'_>>, Self::Error>;

    /// Applies a function over the elements of the iterator in reverse order, producing a single final value.
    #[inline]
    fn rfold<B, F>(mut self, init: B, f: F) -> Result<B, Self::Error>
    where
        Self: Sized,
        F: FnMut(B, Self::Item<'_>) -> Result<B, Self::Error>,
    {
        self.try_rfold(init, f)
    }

    /// Applies a function over the elements of the iterator in reverse, producing a single final value.
    ///
    /// This is used as the "base" of many methods on `DoubleEndedFallibleLendingIterator`.
    #[inline]
    fn try_rfold<B, E, F>(&mut self, mut init: B, mut f: F) -> Result<B, E>
    where
        Self: Sized,
        E: From<Self::Error>,
        F: FnMut(B, Self::Item<'_>) -> Result<B, E>,
    {
        while let Some(v) = self.next_back()? {
            init = f(init, v)?;
        }
        Ok(init)
    }
}

/// An iterator which applies a fallible transform to the elements of the
/// underlying iterator.
#[derive(Clone, Debug)]
pub struct Map<T, F> {
    pub it: T,
    pub f: F,
}

/// An iterator which yields the elements of one iterator followed by another.
#[derive(Clone, Debug)]
pub struct Chain<T, U> {
    front: T,
    back: U,
    state: ChainState,
}

/// An iterator that yields the iteration count as well as the values of the
/// underlying iterator.
#[derive(Clone, Debug)]
pub struct Enumerate<I> {
    it: I,
    n: Cell<usize>,
}

impl<I> FallibleLendingIterator for Enumerate<I>
where
    I: FallibleLendingIterator,
{
    type Error = I::Error;
    type Item<'a> = (usize, I::Item<'a>);

    fn next<'a>(&'a mut self) -> Result<Option<Self::Item<'a>>, Self::Error> {
        let next = self.it.next()?;
        match next {
            Some(r) => {
                let i = self.n.get();
                self.n.set(i + 1);
                Ok(Some((i, r)))
            }
            None => Ok(None),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

/// An iterator which uses a fallible predicate to determine which values of the
/// underlying iterator should be yielded.
#[derive(Clone, Debug)]
pub struct Filter<I, F> {
    pub it: I,
    pub f: F,
}

/// An iterator which both filters and maps the values of the underlying
/// iterator.
#[derive(Clone, Debug)]
pub struct FilterMap<I, F> {
    it: I,
    f: F,
}

impl<B, I, F> FallibleLendingIterator for FilterMap<I, F>
where
    I: FallibleLendingIterator,
    F: FnMut(I::Item<'_>) -> Result<Option<B>, I::Error>,
{
    type Error = I::Error;
    type Item<'a> = B;

    #[inline]
    fn next(&mut self) -> Result<Option<B>, I::Error> {
        let map = &mut self.f;
        self.it
            .try_fold((), |(), v| match map(v)? {
                Some(v) => Err(FoldStop::Break(Some(v))),
                None => Ok(()),
            })
            .map(|()| None)
            .unpack_fold()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.it.size_hint().1)
    }

    #[inline]
    fn try_fold<C, E, G>(&mut self, init: C, mut f: G) -> Result<C, E>
    where
        E: From<I::Error>,
        G: FnMut(C, B) -> Result<C, E>,
    {
        let map = &mut self.f;
        self.it.try_fold(init, |acc, v| match map(v)? {
            Some(v) => f(acc, v),
            None => Ok(acc),
        })
    }
}

impl<B, I, F> DoubleEndedFallibleLendingIterator for FilterMap<I, F>
where
    I: DoubleEndedFallibleLendingIterator,
    F: FnMut(I::Item<'_>) -> Result<Option<B>, I::Error>,
{
    #[inline]
    fn next_back(&mut self) -> Result<Option<B>, I::Error> {
        let map = &mut self.f;
        self.it
            .try_rfold((), |(), v| match map(v)? {
                Some(v) => Err(FoldStop::Break(Some(v))),
                None => Ok(()),
            })
            .map(|()| None)
            .unpack_fold()
    }

    #[inline]
    fn try_rfold<C, E, G>(&mut self, init: C, mut f: G) -> Result<C, E>
    where
        E: From<I::Error>,
        G: FnMut(C, B) -> Result<C, E>,
    {
        let map = &mut self.f;
        self.it.try_rfold(init, |acc, v| match map(v)? {
            Some(v) => f(acc, v),
            None => Ok(acc),
        })
    }
}

/// An iterator which flattens an iterator of iterators, yielding those iterators' elements.

/// An iterator that yields `Ok(None)` forever after the underlying iterator
/// yields `Ok(None)` once.
#[derive(Clone, Debug)]
pub struct Fuse<I> {
    it: I,
    done: bool,
}

impl<I> FallibleLendingIterator for Fuse<I>
where
    I: FallibleLendingIterator,
{
    type Error = I::Error;
    type Item<'a> = I::Item<'a>;

    #[inline]
    fn next<'a>(&'a mut self) -> Result<Option<Self::Item<'a>>, Self::Error> {
        if self.done {
            return Ok(None);
        }

        match self.it.next()? {
            Some(i) => Ok(Some(i)),
            None => {
                self.done = true;
                Ok(None)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done { (0, Some(0)) } else { self.it.size_hint() }
    }

    #[inline]
    fn try_fold<B, E, F>(&mut self, init: B, f: F) -> Result<B, E>
    where
        E: From<I::Error>,
        F: FnMut(B, I::Item<'_>) -> Result<B, E>,
    {
        if self.done { Ok(init) } else { self.it.try_fold(init, f) }
    }
}

/// An iterator which passes each element to a closure before returning it.
#[derive(Clone, Debug)]
pub struct Inspect<I, F> {
    it: I,
    f: F,
}

impl<I, F> FallibleLendingIterator for Inspect<I, F>
where
    I: FallibleLendingIterator,
    F: FnMut(&I::Item<'_>) -> Result<(), I::Error>,
{
    type Error = I::Error;
    type Item<'a> = I::Item<'a>;

    #[inline]
    fn next<'a>(&'a mut self) -> Result<Option<Self::Item<'a>>, Self::Error> {
        match self.it.next()? {
            Some(i) => {
                (self.f)(&i)?;
                Ok(Some(i))
            }
            None => Ok(None),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    #[inline]
    fn try_fold<B, E, G>(&mut self, init: B, mut f: G) -> Result<B, E>
    where
        E: From<I::Error>,
        G: FnMut(B, I::Item<'_>) -> Result<B, E>,
    {
        let inspect = &mut self.f;
        self.it.try_fold(init, |acc, v| {
            inspect(&v)?;
            f(acc, v)
        })
    }
}

impl<I, F> DoubleEndedFallibleLendingIterator for Inspect<I, F>
where
    I: DoubleEndedFallibleLendingIterator,
    F: FnMut(&I::Item<'_>) -> Result<(), I::Error>,
{
    #[inline]
    fn next_back(&mut self) -> Result<Option<I::Item<'_>>, I::Error> {
        match self.it.next_back()? {
            Some(i) => {
                (self.f)(&i)?;
                Ok(Some(i))
            }
            None => Ok(None),
        }
    }

    #[inline]
    fn try_rfold<B, E, G>(&mut self, init: B, mut f: G) -> Result<B, E>
    where
        E: From<I::Error>,
        G: FnMut(B, I::Item<'_>) -> Result<B, E>,
    {
        let inspect = &mut self.f;
        self.it.try_rfold(init, |acc, v| {
            inspect(&v)?;
            f(acc, v)
        })
    }
}
/// An iterator which applies a transform to the errors of the underlying
/// iterator.
#[derive(Clone, Debug)]
pub struct MapErr<I, F> {
    it: I,
    f: F,
}

impl<B, F, I> FallibleLendingIterator for MapErr<I, F>
where
    I: FallibleLendingIterator,
    F: FnMut(I::Error) -> B,
{
    type Error = B;
    type Item<'a> = I::Item<'a>;

    #[inline]
    fn next(&mut self) -> Result<Option<I::Item<'_>>, B> {
        self.it.next().map_err(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    #[inline]
    fn try_fold<C, E, G>(&mut self, init: C, mut f: G) -> Result<C, E>
    where
        E: From<B>,
        G: FnMut(C, I::Item<'_>) -> Result<C, E>,
    {
        self.it.try_fold(init, |acc, v| f(acc, v).map_err(MappedErr::Fold)).map_err(|e| match e {
            MappedErr::It(e) => (self.f)(e).into(),
            MappedErr::Fold(e) => e,
        })
    }
}

impl<B, F, I> DoubleEndedFallibleLendingIterator for MapErr<I, F>
where
    I: DoubleEndedFallibleLendingIterator,
    F: FnMut(I::Error) -> B,
{
    #[inline]
    fn next_back(&mut self) -> Result<Option<I::Item<'_>>, B> {
        self.it.next_back().map_err(&mut self.f)
    }

    #[inline]
    fn try_rfold<C, E, G>(&mut self, init: C, mut f: G) -> Result<C, E>
    where
        E: From<B>,
        G: FnMut(C, I::Item<'_>) -> Result<C, E>,
    {
        self.it.try_rfold(init, |acc, v| f(acc, v).map_err(MappedErr::Fold)).map_err(|e| match e {
            MappedErr::It(e) => (self.f)(e).into(),
            MappedErr::Fold(e) => e,
        })
    }
}

enum MappedErr<T, U> {
    It(T),
    Fold(U),
}

impl<T, U> From<T> for MappedErr<T, U> {
    #[inline]
    fn from(t: T) -> MappedErr<T, U> {
        MappedErr::It(t)
    }
}

/// An iterator which yields elements of the underlying iterator in reverse
/// order.
#[derive(Clone, Debug)]
pub struct Rev<I>(I);

impl<I> FallibleLendingIterator for Rev<I>
where
    I: DoubleEndedFallibleLendingIterator,
{
    type Error = I::Error;
    type Item<'a> = I::Item<'a>;

    #[inline]
    fn next(&mut self) -> Result<Option<I::Item<'_>>, I::Error> {
        self.0.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn try_fold<B, E, F>(&mut self, init: B, f: F) -> Result<B, E>
    where
        E: From<I::Error>,
        F: FnMut(B, I::Item<'_>) -> Result<B, E>,
    {
        self.0.try_rfold(init, f)
    }
}

impl<I> DoubleEndedFallibleLendingIterator for Rev<I>
where
    I: DoubleEndedFallibleLendingIterator,
{
    #[inline]
    fn next_back(&mut self) -> Result<Option<I::Item<'_>>, I::Error> {
        self.0.next()
    }

    #[inline]
    fn try_rfold<B, E, F>(&mut self, init: B, f: F) -> Result<B, E>
    where
        E: From<I::Error>,
        F: FnMut(B, I::Item<'_>) -> Result<B, E>,
    {
        self.0.try_fold(init, f)
    }
}

/// An iterator which applies a stateful closure.
#[derive(Clone, Debug)]
pub struct Scan<I, St, F> {
    it: I,
    f: F,
    state: St,
}

impl<B, I, St, F> FallibleLendingIterator for Scan<I, St, F>
where
    I: FallibleLendingIterator,
    F: FnMut(&mut St, I::Item<'_>) -> Result<Option<B>, I::Error>,
{
    type Error = I::Error;
    type Item<'a> = B;

    #[inline]
    fn next(&mut self) -> Result<Option<B>, I::Error> {
        match self.it.next()? {
            Some(v) => (self.f)(&mut self.state, v),
            None => Ok(None),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.it.size_hint();
        (0, hint.1)
    }
}

/// An iterator that yields pairs of this iterator's and another iterator's
/// values.
#[derive(Clone, Debug)]
pub struct Zip<T, U>(T, U);

impl<T, U> FallibleLendingIterator for Zip<T, U>
where
    T: FallibleLendingIterator,
    U: FallibleLendingIterator<Error = T::Error>,
{
    type Error = T::Error;
    type Item<'a> = (T::Item<'a>, U::Item<'a>);

    #[inline]
    fn next<'a>(&'a mut self) -> Result<Option<Self::Item<'a>>, Self::Error> {
        match (self.0.next()?, self.1.next()?) {
            (Some(a), Some(b)) => Ok(Some((a, b))),
            _ => Ok(None),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let a = self.0.size_hint();
        let b = self.1.size_hint();

        let low = cmp::min(a.0, b.0);

        let high = match (a.1, b.1) {
            (Some(a), Some(b)) => Some(cmp::min(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        (low, high)
    }
}
