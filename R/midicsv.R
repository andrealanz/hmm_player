midicsv <- function(input, exe = "src/midicsv/Midicsv.exe", output = "test.csv"){
  string <- paste(exe, input, output)
  system(string) 
}