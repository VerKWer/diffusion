use std::{fmt::{self, Display}, marker::PhantomData, mem::MaybeUninit};

use rand::Rng;
use serde::{
	de::{self, SeqAccess, Visitor},
	ser::SerializeTuple,
	Deserialize, Serialize, Serializer,
};

use crate::{diffusion::DiffusionFunc, evaluation::Evaluator, globals::GENERATION_SIZE};

#[derive(Debug)]
pub struct Generation<F, E> {
	pub members: [E; GENERATION_SIZE as usize],
	_marker: PhantomData<F>
}

impl<F, E> Generation<F, E> {
	#[inline(always)]
	pub fn new(members: [E; GENERATION_SIZE as usize]) -> Self {
		Self { members, _marker: PhantomData }
	}
}

impl<F, E: Display> Display for Generation<F, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}", self.members[0])?;
		for i in 1..GENERATION_SIZE {
			write!(f, "\n {}", self.members[i as usize])?;
		}
		write!(f, "]")

    }
}

impl<F: DiffusionFunc, E: Evaluator<F>> Generation<F, E> {
	pub fn random(rng: &mut impl Rng) -> Self {
		let mut arr: [MaybeUninit<E>; GENERATION_SIZE as usize] = unsafe { MaybeUninit::uninit().assume_init() };
		for cand in &mut arr {
			*cand = MaybeUninit::new(E::random(rng));
		}
		// let arr = unsafe { mem::transmute(arr) };  // incompatible with const generics
		let ptr = &mut arr as *mut _ as *mut [E; GENERATION_SIZE as usize];
		let members = unsafe { ptr.read() };
		core::mem::forget(arr);
		Self::new(members)
	}
}

impl<F, E: Serialize> Serialize for Generation<F, E> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where S: Serializer {
		let mut s = serializer.serialize_tuple(GENERATION_SIZE as usize)?;
		for item in self.members.iter() {
			s.serialize_element(item)?;
		}
		s.end()
	}
}

struct GenerationVisitor<F, E> {
	_marker1: PhantomData<F>,
	_marker2: PhantomData<E>,
}

impl<F, E> GenerationVisitor<F, E> {
	pub fn new() -> Self { GenerationVisitor { _marker1: PhantomData, _marker2: PhantomData } }
}

impl<'de, F, E> Visitor<'de> for GenerationVisitor<F, E>
where E: Deserialize<'de>
{
	type Value = Generation<F, E>;

	fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
		write!(formatter, "an array of size {}", GENERATION_SIZE)
	}

	fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
	where A: SeqAccess<'de> {
		let mut arr: [MaybeUninit<E>; GENERATION_SIZE as usize] = unsafe { MaybeUninit::uninit().assume_init() };

		let mut it = arr.iter_mut();
		let mut len = 0_u32;
		let err = loop {
			match (seq.next_element(), it.next()) {
				(Ok(Some(val)), Some(off)) => *off = MaybeUninit::new(val),
				// No error
				(Ok(None), None) => break None,
				// serde error
				(Err(e), _) => break Some(e),
				// Wrong length
				(Ok(None), Some(_)) | (Ok(Some(_)), None) => {
					break Some(de::Error::invalid_length(len as usize, &self))
				}
			}
			len += 1;
		};
		if let Some(err) = err {
			if std::mem::needs_drop::<E>() {
				for elem in std::array::IntoIter::new(arr).take(len as usize) {
                    // Safe because we did initialise `len` many elements.
					unsafe { elem.assume_init(); }
				}
			}
			return Err(err);
		}

		let ptr = &mut arr as *mut _ as *mut Generation<F, E>;
		let result = unsafe { ptr.read() };
		std::mem::forget(arr);

		Ok(result)
	}
}

impl<'de, F, E: Deserialize<'de>> Deserialize<'de> for Generation<F, E> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where D: serde::Deserializer<'de> {
		deserializer.deserialize_tuple(GENERATION_SIZE as usize, GenerationVisitor::new())
	}
}
