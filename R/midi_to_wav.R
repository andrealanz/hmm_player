midi_to_wav <- function(input, output = 'www/output.wav'){
  system2("fluidsynth", 
          args = c('-F', output, 'src/fluidsynth/general.sf2', input))
}

