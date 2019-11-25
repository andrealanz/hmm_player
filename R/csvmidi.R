csvmidi <- function(input, exe = "src/midicsv/Csvmidi.exe", output = "test.mid"){
  string <- paste(exe, input, output)
  system(string) 
}