generate_midi <- function(input, output = "myMidi.mid"){
  #convert midi to csv
  midicsv(input, output = "output.csv")
  hmm_compose("output.csv", 
              'HMM_output.csv', 256, 
              'first_order', 2, 10E-7, 1, metrics_calc = FALSE, case_study = FALSE)
  files = list.files()
  new_csv = files[unlist(lapply(files, FUN = str_detect, pattern = "HMM"))]
  csvmidi(input = new_csv, output = output)
}