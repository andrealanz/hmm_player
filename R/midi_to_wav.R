midi_to_wav <- function(input, output = 'www/output.wav'){
  system2("src/fluidsynth/fluidsynth.exe", 
          args = c('-F', output, 'src/fluidsynth/general.sf2', input))
}

