use futures::executor::block_on;
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

mod macros;
mod matrix;
mod onnx;

median::external! {
    #[name="dice"]
    pub struct MaxExtern {
        onnx_provider: onnx::OnnxProvider,
        threshold: Float64,
        noise_level: Float64,
        seed: Int64,
        list_out: OutList,
    }

    impl MaxObjWrapped<MaxExtern> for MaxExtern {
        fn new(builder: &mut dyn MaxWrappedBuilder<Self>) -> Self {
            let mut onnx_provider = onnx::OnnxProvider::new();
            block_on(async {
                handle_result!(onnx_provider.init().await);
            });

            Self {
                onnx_provider,
                threshold: Float64::new(0.5),
                noise_level: Float64::new(0.2),
                seed: Int64::new(0),
                list_out: builder.add_list_outlet_with_assist("list outlet"),
            }
        }

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

        #[list]
        pub fn list(&self, atoms: &[Atom]) {
            let coo_input = atoms.iter().map(|atom| { atom.get_int() as usize }).collect();

            let flat_input = handle_result!(matrix::coo_to_flat(coo_input, 16, 16));
            let flat_flipped = handle_result!(matrix::flat_horizontal_flip(flat_input, 16, 16));
            let noisy_flat_input = matrix::apply_noise(flat_flipped, self.noise_level.get() as f32, self.seed.get() as u64);

            block_on(async {
                handle_result!(self.onnx_provider.run(noisy_flat_input, |output| {
                    let flat_output = matrix::apply_threshold(output.to_vec(), self.threshold.get() as f32);
                    let flat_flipped = handle_result!(matrix::flat_horizontal_flip(flat_output, 16, 16));

                    let coo_output = handle_result!(matrix::flat_to_coo(flat_flipped, 16, 16));
                    let atom_outputs = coo_output.iter()
                                    .map(|&x| x as isize)
                                    .map(Atom::from)
                                    .collect::<Vec<Atom>>();

                    let _ = self.list_out.send(&atom_outputs);
                }).await);
            });
        }
    }
}
