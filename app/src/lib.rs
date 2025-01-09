use median::{
    atom::Atom,
    attr::{AttrBuilder, AttrType},
    builder::MaxWrappedBuilder,
    class::Class,
    num::Float64,
    num::Int64,
    outlet::OutList,
    post,
    wrapper::{attr_get_tramp, attr_set_tramp, MaxObjWrapped, MaxObjWrapper},
};

use futures::executor::block_on;
use maplit::hashmap;
use std::borrow::Cow;
use wonnx::{utils::InputTensor, utils::OutputTensor, Session};

mod macros;
mod matrix;

const ONNX_MODEL: &[u8] =
    include_bytes!("../../dist/models/default-att_unet-mse_poly_penalty.onnx");

// Wrap your external in this macro to get the system to register your object and
// automatically generate trampolines and related functionalities.
median::external! {
    #[name="dice"]
    pub struct MaxExtern {
        threshold: Float64,
        noise_level: Float64,
        seed: Int64,
        list_out: OutList, // Add an outlet field
        session: Session,
    }

    // Implement the MaxObjWrapped trait
    impl MaxObjWrapped<MaxExtern> for MaxExtern {
        // Create an instance of your object, and set up inlets/outlets and clocks
        fn new(builder: &mut dyn MaxWrappedBuilder<Self>) -> Self {
            let mut session: Option<Session> = None;
            block_on(async {
                session = Session::from_bytes(ONNX_MODEL).await.ok();
            });

            Self {
                threshold: Float64::new(0.5),
                noise_level: Float64::new(0.2),
                seed: Int64::new(0),
                list_out: builder.add_list_outlet_with_assist("list outlet"), // Initialize outlet
                session: session.unwrap(),
            }
        }

        // Register any methods you need for your class
        fn class_setup(c: &mut Class<MaxObjWrapper<Self>>) {
            c.add_attribute(
                AttrBuilder::new_accessors(
                    "threshold",
                    AttrType::Float64,
                    Self::threshold_tramp,
                    Self::set_threshold_tramp,
                )
                .build()
                .unwrap(),
            )
            .expect("failed to add attribute");

            c.add_attribute(
                AttrBuilder::new_accessors(
                    "noiseLevel",
                    AttrType::Float64,
                    Self::noise_level_tramp,
                    Self::set_noise_level_tramp,
                )
                .build()
                .unwrap(),
            )
            .expect("failed to add attribute");

            c.add_attribute(
                AttrBuilder::new_accessors(
                    "seed",
                    AttrType::Int64,
                    Self::seed_tramp,
                    Self::set_seed_tramp,
                )
                .build()
                .unwrap(),
            )
            .expect("failed to add attribute");
        }
    }

    // Implement additional methods for your object
    impl MaxExtern {
        #[attr_get_tramp]
        pub fn threshold(&self) -> f64 {
            self.threshold.get()
        }

        #[attr_set_tramp]
        pub fn set_threshold(&self, v: f64) {
            self.threshold.set(v);
        }

        #[attr_get_tramp]
        pub fn noise_level(&self) -> f64 {
            self.noise_level.get()
        }

        #[attr_set_tramp]
        pub fn set_noise_level(&self, v: f64) {
            self.noise_level.set(v);
        }

        #[attr_get_tramp]
        pub fn seed(&self) -> isize {
            self.seed.get()
        }

        #[attr_set_tramp]
        pub fn set_seed(&self, v: isize) {
            self.seed.set(v);
        }

        // List method to process and forward the received list to the outlet
        #[list]
        pub fn list(&self, atoms: &[Atom]) {
            let coo_input = atoms.iter().map(|atom| { atom.get_int() as usize }).collect();
            let flat_input = handle_result!(matrix::coo_to_flat(coo_input, 16, 16));
            let flat_flipped = handle_result!(matrix::flat_horizontal_flip(flat_input, 16, 16));

            let noisy_flat_input = matrix::apply_noise(flat_flipped, self.noise_level.get() as f32, self.seed.get() as u64);
            let input_map = hashmap! {
                "input".to_string() => InputTensor::F32(Cow::Owned(noisy_flat_input)),
            };

            block_on(async {
                let output_map = handle_result!(self.session.run(&input_map).await);
                let output_tensor = handle_option!(output_map.get("output"), "Missing ONNX output tensor");

                match output_tensor {
                    OutputTensor::F32(tensor) => {
                        let flat_output = matrix::apply_threshold(tensor.to_vec(), self.threshold.get() as f32);
                        let flat_flipped = handle_result!(matrix::flat_horizontal_flip(flat_output, 16, 16));
                        let coo_output = handle_result!(matrix::flat_to_coo(flat_flipped, 16, 16));
                        let atom_outputs = coo_output.iter()
                                        .map(|&x| x as isize)
                                        .map(Atom::from)
                                        .collect::<Vec<Atom>>();
                        let _ = self.list_out.send(&atom_outputs);
                    }
                    _ => {
                        post!("Unexpected tensor type");
                    }
                }
            });
        }
    }
}
