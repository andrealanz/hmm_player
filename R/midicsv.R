midicsv <- function(input, exe = "src/midicsv-linux/midicsv", output = "test.csv"){
  string <- paste(exe, input, output)
  system(string) 
}
