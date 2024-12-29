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

// Wrap your external in this macro to get the system to register your object and
// automatically generate trampolines and related functionalities.
median::external! {
    #[name="dice"]
    pub struct MaxExtern {
        fvalue: Float64,
        list_out: OutList, // Add an outlet field
    }

    // Implement the MaxObjWrapped trait
    impl MaxObjWrapped<MaxExtern> for MaxExtern {
        // Create an instance of your object, and set up inlets/outlets and clocks
        fn new(builder: &mut dyn MaxWrappedBuilder<Self>) -> Self {
            Self {
                fvalue: Float64::new(0.0),
                list_out: builder.add_list_outlet_with_assist("list outlet"), // Initialize outlet
            }
        }

        // Register any methods you need for your class
        fn class_setup(c: &mut Class<MaxObjWrapper<Self>>) {
            c.add_attribute(
                AttrBuilder::new_accessors(
                    "foo",
                    AttrType::Float64,
                    Self::foo_tramp,
                    Self::set_foo_tramp,
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
        pub fn foo(&self) -> f64 {
            self.fvalue.get()
        }

        // Float attribute setter trampoline
        #[attr_set_tramp]
        pub fn set_foo(&self, v: f64) {
            self.fvalue.set(v);
        }

        // List method to process and forward the received list to the outlet
        #[list]
        pub fn list(&self, atoms: &[Atom]) {
            let expected_size = 16 * 16;
            if atoms.len() != expected_size {
                post!(
                    "Error: Expected 256 elements (16x16 matrix), but got {} elements.",
                    atoms.len()
                );
                return;
            }

            let mut matrix = [[0.0; 16]; 16];
            for (i, atom) in atoms.iter().enumerate() {
                if let Some(_value) = atom.get_value() {
                    matrix[i / 16][i % 16] = atom.get_float();
                } else {
                    post!("Error: Element {} is not a valid float.", i);
                    return;
                }
            }

            post!("Successfully received a 16x16 matrix.");
            let _ = self.list_out.send(atoms);
        }
    }
}
