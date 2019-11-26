library(reticulate)
library(stringr)
source("R/midicsv.R")
source("R/csvmidi.R")

use_python("C:\\Users\\lanza\\Anaconda3\\python.exe")

home_dir = getwd()
setwd("src/hmm")
source_python("hmm.py")
setwd(home_dir)

source("R/midi_to_wav.R")
source("R/generate_HMM.R")

