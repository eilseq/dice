inlets = 1;
outlets = 2;

function dictionary(dictName) {
  const DICE_TENSOR_INITIAL_INDEX = 1;
  const DICE_TIME_SIGNATURE = [4, 4];
  const ABLETON_DRUM_RACK_INITIAL_PITCH = 36;

  const dict = new Dict(dictName);
  const coo = new Array();
  const notesToRemove = new Array();

  dict.get("notes").forEach(function (note, index) {
    const x = note.get("start_time") * DICE_TIME_SIGNATURE[0];
    const y = note.get("pitch") - ABLETON_DRUM_RACK_INITIAL_PITCH;

    if (x >= 0 && x < 16) {
      if (y >= 0 && y < 16) {
        coo.push(Math.round(x) + DICE_TENSOR_INITIAL_INDEX);
        coo.push(Math.round(y) + DICE_TENSOR_INITIAL_INDEX);
        notesToRemove.push(index);
      }
    }
  });

  notesToRemove.forEach(function (indexNoteToRemove) {
    dict.get("notes").splice(indexNoteToRemove, 1);
  });

  outlet(1, coo);
  outlet(0, "dictionary", dictName);
}
