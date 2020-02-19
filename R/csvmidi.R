csvmidi <- function(input, exe = "src/midicsv-linux/csvmidi", output = "test.mid"){
  string <- paste(exe, input, output)
  system(string) 
}
