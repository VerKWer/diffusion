use std::{fmt, marker::PhantomData, mem::MaybeUninit};

use rand::Rng;
use serde::{
	de::{self, SeqAccess, Visitor},
	ser::SerializeTuple,
	Deserialize, Serialize, Serializer,
};

use crate::{diffusion::DiffusionFunc, globals::GENERATION_SIZE};

#[derive(Debug)]
pub struct Generation<F>(pub [F; GENERATION_SIZE as usize]);

impl<F: DiffusionFunc> Generation<F> {
	pub fn random(rng: &mut impl Rng) -> Self {
		let mut arr: [MaybeUninit<F>; GENERATION_SIZE as usize] = unsafe { MaybeUninit::uninit().assume_init() };
		for cand in &mut arr {
			*cand = MaybeUninit::new(F::random(rng));
		}
		// let arr = unsafe { mem::transmute(arr) };  // incompatible with const generics
		let ptr = &mut arr as *mut _ as *mut [F; GENERATION_SIZE as usize];
		let members = unsafe { ptr.read() };
		core::mem::forget(arr);
		Self(members)
	}
}

impl<F: Serialize> Serialize for Generation<F> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where S: Serializer {
		let mut s = serializer.serialize_tuple(GENERATION_SIZE as usize)?;
		for item in self.0.iter() {
			s.serialize_element(item)?;
		}
		s.end()
	}
}

struct GenerationVisitor<F> {
	_marker: PhantomData<F>,
}

impl<F> GenerationVisitor<F> {
	pub fn new() -> Self { GenerationVisitor { _marker: PhantomData } }
}

impl<'de, F> Visitor<'de> for GenerationVisitor<F>
where F: Deserialize<'de>
{
	type Value = Generation<F>;

	fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
		write!(formatter, "an array of size {}", GENERATION_SIZE)
	}

	fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
	where A: SeqAccess<'de> {
		let mut arr: [MaybeUninit<F>; GENERATION_SIZE as usize] = unsafe { MaybeUninit::uninit().assume_init() };

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
			if std::mem::needs_drop::<F>() {
				for elem in std::array::IntoIter::new(arr).take(len as usize) {
                    // Safe because we did initialise `len` many elements.
					unsafe { elem.assume_init(); }
				}
			}
			return Err(err);
		}

		let ptr = &mut arr as *mut _ as *mut Generation<F>;
		let result = unsafe { ptr.read() };
		std::mem::forget(arr);

		Ok(result)
	}
}

impl<'de, F: Deserialize<'de>> Deserialize<'de> for Generation<F> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where D: serde::Deserializer<'de> {
		deserializer.deserialize_tuple(GENERATION_SIZE as usize, GenerationVisitor::new())
	}
}
