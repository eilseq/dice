inlets = 1;
outlets = 1;

var coo;

function dictionary(dictName) {
  const DICE_TENSOR_INITIAL_INDEX = 1;
  const DICE_TIME_SIGNATURE = [4, 4];
  const ABLETON_DRUM_RACK_INITIAL_PITCH = 36;

  const dict = new Dict(dictName);

  if (coo) {
    for (var i = 0; i < coo.length; i += 2) {
      if (i + 1 > coo.length) {
        return; // fail silently
      }

      const x = coo[i] - DICE_TENSOR_INITIAL_INDEX;
      const y = coo[i + 1] - DICE_TENSOR_INITIAL_INDEX;

      const start_time = x / DICE_TIME_SIGNATURE[0];
      const pitch = y + ABLETON_DRUM_RACK_INITIAL_PITCH;

      const note = new Dict();
      note.set("start_time", start_time);
      note.set("pitch", pitch);
      note.set("duration", 1 / 4);

      dict.append("notes", note);
    }
  }

  outlet(0, "dictionary", dict.name);
}

function list() {
  coo = arrayfromargs(arguments);
}
