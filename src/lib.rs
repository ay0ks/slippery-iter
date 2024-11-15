#![allow(incomplete_features)]
#![feature(generic_const_exprs, box_patterns)]
use std::{fmt, marker::PhantomData, str::Chars};

#[derive(Clone, Debug)]
pub struct SlipperyIter<'a, I>
where
  I: Iterator + Sized,
  I::Item: Clone + fmt::Debug + 'a,
{
  offset: usize,
  inner: Box<[I::Item]>,
  _marker: PhantomData<&'a [I::Item]>,
}

#[derive(Clone, Debug)]
pub struct SlipperyCharsIter<'a> {
  inner: SlipperyIter<'a, Chars<'a>>,
  pub line: usize,
  pub column: usize,
  line_queue: Vec<usize>,
  column_queue: Vec<usize>,
}

impl<I> From<I> for SlipperyIter<'_, I>
where
  I: Iterator + Clone,
  I::Item: Clone + fmt::Debug,
{
  fn from(iter: I) -> Self {
    Self {
      offset: 0,
      inner: iter.clone().collect::<Vec<_>>().into(),
      _marker: PhantomData,
    }
  }
}

impl<'a> From<Chars<'a>> for SlipperyCharsIter<'a> {
  fn from(iter: Chars<'a>) -> Self {
    let mut self_ = Self {
      inner: iter.slippery(),
      line: 0,
      column: 0,
      line_queue: Vec::new(),
      column_queue: Vec::new(),
    };
    self_.line_queue.reserve(16);
    self_.column_queue.reserve(16);
    self_
  }
}

pub trait IntoSlipperyIterator<'a>: IntoIterator {
  type Item;
  type IntoSlipperyIter: Iterator<Item = <Self as IntoSlipperyIterator<'a>>::Item> + 'a;

  fn into_slippery_iter(self) -> Self::IntoSlipperyIter;
}

pub trait IntoSlipperyCharsIterator<'a>: IntoIterator {
  fn into_slippery_chars_iter(self) -> SlipperyCharsIter<'a>;
}

impl<'a, I: 'a, T> IntoSlipperyIterator<'a> for T
where
  T: IntoIterator<Item = I> + Clone + 'a,
  T::Item: Clone + fmt::Debug,
  T::IntoIter: Clone,
{
  type Item = T::Item;
  type IntoSlipperyIter = SlipperyIter<'a, T::IntoIter>;

  fn into_slippery_iter(self) -> Self::IntoSlipperyIter {
    SlipperyIter::<'a>::from(self.into_iter())
  }
}

impl<'a> IntoSlipperyCharsIterator<'a> for Chars<'a> {
  fn into_slippery_chars_iter(self) -> SlipperyCharsIter<'a> {
    SlipperyCharsIter::<'a>::from(self)
  }
}

pub trait SlipperyIterator<'a>: Iterator
where
  Self: Sized,
  <Self as Iterator>::Item: Clone + fmt::Debug,
{
  type Item;

  fn slippery(self) -> SlipperyIter<'a, Self>;
}

#[allow(dead_code)]
pub trait SlipperyCharsIterator<'a> {
  fn slippery_chars(self) -> SlipperyCharsIter<'a>;
}

impl<'a, T> SlipperyIterator<'a> for T
where
  T: Iterator + Clone + Sized + 'a,
  T::Item: Clone + fmt::Debug,
{
  type Item = T::Item;

  fn slippery(self) -> SlipperyIter<'a, Self> {
    self.into_slippery_iter()
  }
}

impl<'a> SlipperyCharsIterator<'a> for &'a str {
  fn slippery_chars(self) -> SlipperyCharsIter<'a> {
    self.chars().into_slippery_chars_iter()
  }
}

impl<I, T> Iterator for SlipperyIter<'_, I>
where
  I: Iterator<Item = T> + Clone,
  I::Item: Clone + fmt::Debug,
{
  type Item = <I as Iterator>::Item;

  fn next(&mut self) -> Option<<Self as Iterator>::Item> {
    self.consume()
  }
}

impl Iterator for SlipperyCharsIter<'_> {
  type Item = char;

  fn next(&mut self) -> Option<char> {
    match self.consume() {
      '\0' => None,
      x => Some(x),
    }
  }
}

#[allow(dead_code)]
impl<I, T> SlipperyIter<'_, I>
where
  I: Iterator<Item = T>,
  I::Item: Clone + fmt::Debug,
{
  pub fn peek_forward(&self) -> Option<I::Item> {
    self.peek_forward_many::<1>().first()?.clone()
  }

  pub fn peek_forward_many<const AMOUNT: usize>(&self) -> Box<[Option<I::Item>]> {
    let (mut v, mut o) = (Vec::new(), self.offset);
    for _ in 0..AMOUNT {
      if o >= self.inner.len() || o >= usize::MAX - 1 {
        v.push(None);
      } else {
        v.push(Some(self.inner[o].clone()));
        o = o.saturating_add(1);
      }
    }
    v.into_boxed_slice()
  }

  pub fn peek_backward(&self) -> Option<I::Item> {
    self.peek_backward_many::<1>().first()?.clone()
  }

  pub fn peek_backward_many<const AMOUNT: usize>(&self) -> Box<[Option<I::Item>]> {
    let (mut v, mut o) = (Vec::new(), self.offset.saturating_sub(1));
    for _ in 0..AMOUNT {
      let x = &self.inner[o];
      o = o.saturating_sub(1);
      v.push(Some(x.clone()));
    }
    v.into_boxed_slice()
  }

  fn look_around<const AMOUNT: usize, const L: isize, const DIR: bool>(
    &self,
  ) -> Box<[Option<I::Item>]>
  where
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    match L.signum() {
      -1 => match DIR {
        true => {
          let mut v = self.peek_backward_many::<{ L.unsigned_abs() }>().to_vec();
          v.extend(self.peek_forward_many::<AMOUNT>());
          v.into_boxed_slice()
        }
        false => self.peek_backward_many::<{ AMOUNT + L.unsigned_abs() }>(),
      },
      1 => match DIR {
        false => {
          let mut v = self.peek_backward_many::<{ L.unsigned_abs() }>().to_vec();
          v.extend(self.peek_forward_many::<AMOUNT>());
          v.into_boxed_slice()
        }
        true => self.peek_forward_many::<{ AMOUNT + L.unsigned_abs() }>(),
      },
      0 | _ => match DIR {
        true => self.peek_forward_many::<AMOUNT>(),
        false => self.peek_backward_many::<AMOUNT>(),
      },
    }
  }

  pub fn consume(&mut self) -> Option<I::Item> {
    self.consume_many::<1, 0>().first()?.clone()
  }

  pub fn consume_if<F>(&mut self, predicate: F) -> Option<I::Item>
  where
    F: FnOnce(Option<I::Item>) -> bool,
  {
    self
      .consume_many_if::<_, 1, 0>(|mut v| predicate(v.remove(0)))
      .first()?
      .clone()
  }

  pub fn consume_if_then<F1, F2>(&mut self, predicate: F1, action: F2) -> Option<I::Item>
  where
    F1: FnOnce(Option<I::Item>) -> bool,
    F2: FnOnce(Option<I::Item>) -> Option<I::Item>,
  {
    self.consume_if_then_else::<F1, F2, _>(predicate, action, || {})
  }

  pub fn consume_if_then_else<F1, F2, F3>(
    &mut self,
    predicate: F1,
    action: F2,
    action_else: F3,
  ) -> Option<I::Item>
  where
    F1: FnOnce(Option<I::Item>) -> bool,
    F2: FnOnce(Option<I::Item>) -> Option<I::Item>,
    F3: FnOnce(),
  {
    self
      .consume_many_if_then_else::<_, _, _, 1, 0>(
        |v| {
          predicate(match v.first() {
            Some(&None) | None => None,
            Some(Some(value)) => Some(value.clone()),
          })
        },
        |bv| Box::new([action(bv.first().unwrap().clone())]),
        action_else,
      )
      .first()?
      .clone()
  }

  pub fn consume_many<const AMOUNT: usize, const L: isize>(&mut self) -> Box<[Option<I::Item>]>
  where
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    let v = self.look_around::<AMOUNT, L, true>();
    self.offset = self.offset.saturating_add(AMOUNT);
    v
  }

  pub fn consume_many_if<F, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F,
  ) -> Box<[Option<I::Item>]>
  where
    F: FnOnce(Vec<Option<I::Item>>) -> bool,
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    if predicate(self.look_around::<AMOUNT, L, true>().to_vec()) {
      self.consume_many::<AMOUNT, L>()
    } else {
      Box::new([])
    }
  }

  pub fn consume_many_if_then<F1, F2, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F1,
    action: F2,
  ) -> Box<[Option<I::Item>]>
  where
    F1: FnOnce(Vec<Option<I::Item>>) -> bool,
    F2: FnOnce(Box<[Option<I::Item>]>) -> Box<[Option<I::Item>]>,
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    self.consume_many_if_then_else::<F1, F2, _, AMOUNT, L>(predicate, action, || {})
  }

  pub fn consume_many_if_then_else<F1, F2, F3, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F1,
    action: F2,
    action_else: F3,
  ) -> Box<[Option<I::Item>]>
  where
    F1: FnOnce(Vec<Option<I::Item>>) -> bool,
    F2: FnOnce(Box<[Option<I::Item>]>) -> Box<[Option<I::Item>]>,
    F3: FnOnce(),
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    match self.consume_many_if::<_, AMOUNT, L>(predicate) {
      x @ box [] => {
        action_else();
        x
      }
      y @ box [..] => action(y),
    }
  }

  pub fn restore(&mut self) -> Option<I::Item> {
    self.restore_many::<1, 0>().first()?.clone()
  }

  pub fn restore_if<F>(&mut self, predicate: F) -> Option<I::Item>
  where
    F: FnOnce(Option<I::Item>) -> bool,
  {
    self
      .restore_many_if::<_, 1, 0>(|v| {
        predicate(match v.first() {
          Some(&None) | None => None,
          Some(Some(value)) => Some(value.clone()),
        })
      })
      .first()?
      .clone()
  }

  pub fn restore_if_then<F1, F2>(&mut self, predicate: F1, action: F2) -> Option<I::Item>
  where
    F1: FnOnce(Option<I::Item>) -> bool,
    F2: FnOnce(Option<I::Item>) -> Option<I::Item>,
  {
    self.restore_if_then_else::<F1, F2, _>(predicate, action, || {})
  }

  pub fn restore_if_then_else<F1, F2, F3>(
    &mut self,
    predicate: F1,
    action: F2,
    action_else: F3,
  ) -> Option<I::Item>
  where
    F1: FnOnce(Option<I::Item>) -> bool,
    F2: FnOnce(Option<I::Item>) -> Option<I::Item>,
    F3: FnOnce(),
  {
    self
      .restore_many_if_then_else::<_, _, _, 1, 0>(
        |v| {
          predicate(match v.first() {
            Some(&None) | None => None,
            Some(Some(value)) => Some(value.clone()),
          })
        },
        |bv| Box::new([action(bv.first().unwrap().clone())]),
        action_else,
      )
      .first()?
      .clone()
  }

  pub fn restore_many<const AMOUNT: usize, const L: isize>(&mut self) -> Box<[Option<I::Item>]>
  where
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    let v = self.look_around::<AMOUNT, L, false>();
    self.offset = self.offset.saturating_sub(AMOUNT);
    v
  }

  pub fn restore_many_if<F, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F,
  ) -> Box<[Option<I::Item>]>
  where
    F: FnOnce(Vec<Option<I::Item>>) -> bool,
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    if predicate(self.look_around::<AMOUNT, L, false>().to_vec()) {
      self.restore_many::<AMOUNT, L>()
    } else {
      Box::new([])
    }
  }

  pub fn restore_many_if_then<F1, F2, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F1,
    action: F2,
  ) -> Box<[Option<I::Item>]>
  where
    F1: FnOnce(Vec<Option<I::Item>>) -> bool,
    F2: FnOnce(Box<[Option<I::Item>]>) -> Box<[Option<I::Item>]>,
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    self.restore_many_if_then_else::<F1, F2, _, AMOUNT, L>(predicate, action, || {})
  }

  pub fn restore_many_if_then_else<F1, F2, F3, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F1,
    action: F2,
    action_else: F3,
  ) -> Box<[Option<I::Item>]>
  where
    F1: FnOnce(Vec<Option<I::Item>>) -> bool,
    F2: FnOnce(Box<[Option<I::Item>]>) -> Box<[Option<I::Item>]>,
    F3: FnOnce(),
    [(); AMOUNT + L.unsigned_abs()]: Sized,
  {
    match self.restore_many_if::<_, AMOUNT, L>(predicate) {
      x @ box [] => {
        action_else();
        x
      }
      y @ box [..] => action(y),
    }
  }
}

#[allow(dead_code)]
impl SlipperyCharsIter<'_> {
  pub fn position(&self) -> (usize, usize) {
    (self.line, self.column)
  }

  pub fn peek_forward(&self) -> char {
    *self.peek_forward_many::<1>().first().unwrap()
  }

  pub fn peek_forward_many<const AMOUNT: usize>(&self) -> [char; AMOUNT] {
    self
      .inner
      .peek_forward_many::<AMOUNT>()
      .iter()
      .map(|x| x.unwrap_or('\0'))
      .collect::<Vec<_>>()
      .try_into()
      .unwrap()
  }

  pub fn peek_backward(&self) -> char {
    *self.peek_backward_many::<1>().first().unwrap()
  }

  pub fn peek_backward_many<const AMOUNT: usize>(&self) -> [char; AMOUNT] {
    self
      .inner
      .peek_backward_many::<AMOUNT>()
      .iter()
      .map(|x| x.unwrap_or('\0'))
      .collect::<Vec<_>>()
      .try_into()
      .unwrap()
  }

  pub fn consume(&mut self) -> char {
    *self.consume_many::<1, 0>().first().unwrap()
  }

  pub fn consume_if<F>(&mut self, predicate: F) -> char
  where
    F: FnOnce(char) -> bool,
  {
    *self
      .consume_many_if::<_, 1, 0>(|v| {
        let x = *v.first().unwrap();
        predicate(x)
      })
      .first()
      .unwrap()
  }

  pub fn consume_many<const AMOUNT: usize, const L: isize>(
    &mut self,
  ) -> [char; AMOUNT + L.unsigned_abs()] {
    self
      .inner
      .consume_many::<AMOUNT, L>()
      .iter()
      .filter_map(|x| {
        if let Some(x) = x {
          self.column_queue.push(self.column);
          if *x == '\n' {
            self.line_queue.push(self.line);
            self.line += 1;
            self.column = 0;
          } else {
            self.column += 1;
          }
          self.line_queue.truncate(16);
          self.column_queue.truncate(16);
        }
        x.or(Some('\0'))
      })
      .collect::<Vec<_>>()
      .try_into()
      .unwrap()
  }

  pub fn consume_many_if<F, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F,
  ) -> [char; AMOUNT + L.unsigned_abs()]
  where
    F: FnOnce(Vec<char>) -> bool,
  {
    let l = self
      .inner
      .look_around::<AMOUNT, L, true>()
      .iter()
      .map(|x| x.unwrap_or('\0'))
      .collect::<Vec<_>>()
      .into_boxed_slice();
    if predicate(l.to_vec()) {
      self.consume_many::<AMOUNT, L>()
    } else {
      ['\0'; AMOUNT + L.unsigned_abs()]
    }
  }

  pub fn restore(&mut self) -> char {
    *self.restore_many::<1, 0>().first().unwrap()
  }

  pub fn restore_if<F>(&mut self, predicate: F) -> char
  where
    F: FnOnce(char) -> bool,
  {
    *self
      .restore_many_if::<_, 1, 0>(|v| {
        let x = *v.first().unwrap();
        predicate(x)
      })
      .first()
      .unwrap()
  }

  pub fn restore_many<const AMOUNT: usize, const L: isize>(
    &mut self,
  ) -> [char; AMOUNT + L.unsigned_abs()] {
    self
      .inner
      .restore_many::<AMOUNT, L>()
      .iter()
      .filter_map(|x| {
        if let Some(x) = x {
          self.column = self.column_queue.pop().unwrap_or(0);
          if *x == '\n' {
            self.line = self.line_queue.pop().unwrap_or(0);
          }
        }
        x.or(Some('\0'))
      })
      .collect::<Vec<_>>()
      .try_into()
      .unwrap()
  }

  pub fn restore_many_if<F, const AMOUNT: usize, const L: isize>(
    &mut self,
    predicate: F,
  ) -> [char; AMOUNT + L.unsigned_abs()]
  where
    F: FnOnce(Vec<char>) -> bool,
  {
    let l = self
      .inner
      .look_around::<AMOUNT, L, false>()
      .iter()
      .map(|x| x.unwrap_or('\0'))
      .collect::<Vec<_>>()
      .into_boxed_slice();
    if predicate(l.to_vec()) {
      self.restore_many::<AMOUNT, L>()
    } else {
      ['\0'; AMOUNT + L.unsigned_abs()]
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{
    IntoSlipperyCharsIterator, IntoSlipperyIterator, SlipperyCharsIter, SlipperyCharsIterator,
    SlipperyIter, SlipperyIterator,
  };
  use std::ops::RangeInclusive;

  #[test]
  pub fn slippery_iter_construct_and_collect() {
    let orig = 0..=10;
    let orig_iter = orig.clone();

    let s_iter_1: SlipperyIter<'_, RangeInclusive<i32>> = orig_iter.clone().into();
    let s_iter_2 = orig_iter.clone().slippery();
    let s_iter_3 = orig.into_slippery_iter();
    let s_iter_4 = orig_iter.clone().into_slippery_iter();

    assert_eq!(
      s_iter_1.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>(),
    );
    assert_eq!(
      s_iter_2.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>(),
    );
    assert_eq!(
      s_iter_3.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>(),
    );
    assert_eq!(
      s_iter_4.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>(),
    );
  }

  #[test]
  pub fn slippery_iter_peek_forward() {
    let iter = (0..=10).slippery();

    assert_eq!(iter.peek_forward(), Some(0));
    assert_eq!(
      iter.peek_forward_many::<7>().to_vec(),
      [
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6)
      ]
      .to_vec(),
    );
  }

  #[test]
  pub fn slippery_iter_peek_backward() {
    let mut iter = (0..=10).slippery();
    iter.consume_many::<6, 0>();

    assert_eq!(iter.peek_backward(), Some(5));
    assert_eq!(
      iter.peek_backward_many::<6>().to_vec(),
      [Some(5), Some(4), Some(3), Some(2), Some(1), Some(0),].to_vec(),
    );
  }

  #[test]
  pub fn slippery_iter_consume() {
    let mut iter = (0..=10).slippery();

    assert_eq!(iter.consume(), Some(0));
    assert_eq!(iter.peek_forward(), Some(1));
    assert_eq!(
      iter.consume_many::<5, 1>().to_vec(),
      [Some(1), Some(2), Some(3), Some(4), Some(5), Some(6),].to_vec(),
    );
    assert_eq!(iter.peek_forward(), Some(6));
  }

  #[test]
  pub fn slippery_iter_restore() {
    let mut iter = (0..=10).slippery();

    assert_eq!(
      iter.consume_many::<11, 0>().to_vec(),
      [
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        Some(9),
        Some(10),
      ]
      .to_vec()
    );
    assert_eq!(iter.peek_forward(), None);
    assert_eq!(iter.peek_backward(), Some(10));
    assert_eq!(iter.restore(), Some(10));
    assert_eq!(iter.peek_backward(), Some(9));
    assert_eq!(
      iter.restore_many::<5, -1>().to_vec(),
      [Some(9), Some(8), Some(7), Some(6), Some(5), Some(4),].to_vec()
    );
  }

  #[test]
  #[allow(noop_method_call)]
  pub fn slippery_chars_iter_construct_and_collect() {
    let orig = "abcdefghijklmnopqrstuvwxyz";
    let orig_iter = orig.clone().chars();

    let s_iter_1: SlipperyCharsIter<'_> = orig_iter.clone().into();
    let s_iter_2 = orig.clone().slippery_chars();
    let s_iter_3 = orig_iter.clone().into_slippery_chars_iter();

    assert_eq!(
      s_iter_1.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>()
    );
    assert_eq!(
      s_iter_2.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>()
    );
    assert_eq!(
      s_iter_3.collect::<Vec<_>>(),
      orig_iter.clone().collect::<Vec<_>>()
    );
  }

  #[test]
  pub fn slippery_chars_iter_peek_forward() {
    let iter = "abcdefghijklmnopqrstuvwxyz".slippery_chars();

    assert_eq!(iter.peek_forward(), 'a');
    assert_eq!(
      iter.peek_forward_many::<6>().to_vec(),
      ['a', 'b', 'c', 'd', 'e', 'f',].to_vec()
    );
  }

  #[test]
  pub fn slippery_chars_iter_peek_backward() {
    let mut iter = "abcdefghijklmnopqrstuvwxyz".slippery_chars();
    iter.consume_many::<6, 0>();

    assert_eq!(iter.peek_backward(), 'f');
    assert_eq!(
      iter.peek_backward_many::<6>().to_vec(),
      ['f', 'e', 'd', 'c', 'b', 'a',].to_vec()
    );
  }

  #[test]
  pub fn slippery_chars_iter_consume() {
    let mut iter = "abcdefghijklmnopqrstuvwxyz".slippery_chars();

    assert_eq!(iter.consume(), 'a');
    assert_eq!(iter.peek_forward(), 'b');
    assert_eq!(
      iter.consume_many::<5, 1>().to_vec(),
      ['b', 'c', 'd', 'e', 'f', 'g',].to_vec()
    );
    assert_eq!(iter.peek_forward(), 'g');
  }

  #[test]
  pub fn slippery_chars_iter_restore() {
    let mut iter = "abcdefghijklmnopqrstuvwxyz".slippery_chars();

    assert_eq!(
      iter.consume_many::<26, 0>().to_vec(),
      [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      ]
      .to_vec()
    );
    assert_eq!(iter.peek_forward(), '\0');
    assert_eq!(iter.peek_backward(), 'z');
    assert_eq!(iter.restore(), 'z');
    assert_eq!(iter.peek_backward(), 'y');
    assert_eq!(
      iter.restore_many::<5, -1>(),
      ['y', 'x', 'w', 'v', 'u', 't',]
    );
  }
}
