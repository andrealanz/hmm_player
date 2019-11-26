generate_HMM <- function(input, output = "www/mySong.wav"){
  #convert midi to csv
  midicsv(input, output = "output.csv")
  
  if(file.exists("output.csv")){
    hmm_compose("output.csv", 
              'HMM_output.csv', 256, 
              'first_order', 2, 10E-7, 1, metrics_calc = FALSE, case_study = FALSE)
    file.remove("output.csv")
  }
    
  files = list.files()
  new_csv = files[unlist(lapply(files, FUN = str_detect, pattern = "HMM"))]
  if(!identical(new_csv, character(0)) && file.exists(new_csv)){
    csvmidi(input = new_csv, output = "output.mid")
    file.remove(new_csv)
  }
  if(file.exists("output.mid")){  
    midi_to_wav("output.mid", output = output)
    file.remove("output.mid")
  }
}