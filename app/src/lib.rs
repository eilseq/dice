use median::{
    atom::Atom,
    attr::{AttrBuilder, AttrType},
    builder::MaxWrappedBuilder,
    class::Class,
    max_sys::t_atom_long,
    num::Float64,
    object::MaxObj,
    outlet::OutList,
    post,
    wrapper::{attr_get_tramp, attr_set_tramp, MaxObjWrapped, MaxObjWrapper},
};

use futures::executor::block_on;
use maplit::hashmap;
use std::borrow::Cow;
use wonnx::{utils::InputTensor, utils::OutputTensor, Session};

mod utils;

const ONNX_MODEL: &[u8] =
    include_bytes!("../../dist/models/default-att_unet-mse_poly_penalty.onnx");

// Wrap your external in this macro to get the system to register your object and
// automatically generate trampolines and related functionalities.
median::external! {
    #[name="dice"]
    pub struct MaxExtern {
        threshold: Float64,
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
        }
    }

    // Implement additional methods for your object
    impl MaxExtern {
        // Create a "bang" method and automatically register it
        #[bang]
        pub fn bang(&self) {
            let i = median::inlet::Proxy::get_inlet(self.max_obj());
            median::object_post!(self.max_obj(), "bang inlet {}", i);
        }

        // Create an "int" method and automatically register it
        #[int]
        pub fn int(&self, v: t_atom_long) {
            let i = median::inlet::Proxy::get_inlet(self.max_obj());
            post!("int {} inlet {}", v, i);
        }

        // Float attribute getter trampoline
        #[attr_get_tramp]
        pub fn threshold(&self) -> f64 {
            self.threshold.get()
        }

        // Float attribute setter trampoline
        #[attr_set_tramp]
        pub fn set_threshold(&self, v: f64) {
            self.threshold.set(v);
        }

        // List method to process and forward the received list to the outlet
        #[list]
        pub fn list(&self, atoms: &[Atom]) {
            let coo_input: Vec<usize> = atoms.iter().map(|atom| { atom.get_int() as usize }).collect();
            let flat_input = match utils::binary_matrix::coo_to_flat(coo_input, 16, 16) {
                Ok(matrix_flat) => matrix_flat.iter().map(|&x| x as f32).collect(),
                Err(err) => {
                    post!("Dice Error: {}", err);
                    return;
                }
            };

            let input_map = hashmap! {
                "input".to_string() => InputTensor::F32(Cow::Owned(flat_input)),
            };

            block_on(async {
                let output_map = match self.session.run(&input_map).await {
                    Ok(output_map) => output_map,
                    Err(err) => {
                        post!("Dice Error: {}", err);
                        return;
                    }
                };

                let output_tensor = match output_map.get("output") {
                    Some(tensor) => tensor,
                    None => {
                        post!("Missing output tensor");
                        return;
                    }
                };


                match output_tensor {
                    OutputTensor::F32(tensor) => {
                        let flat_output: Vec<usize> = tensor.to_vec().iter()
                                        .map(|&x| if x > (self.threshold.get() as f32) { 1 } else { 0 })
                                        .collect();

                        let coo_output = match utils::binary_matrix::flat_to_coo(flat_output, 16, 16) {
                            Ok(coo_output) => coo_output,
                            Err(err) => {
                                post!("Dice Error: {}", err);
                                return;
                            }
                        };

                        let atom_outputs:Vec<Atom> = coo_output.iter()
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
