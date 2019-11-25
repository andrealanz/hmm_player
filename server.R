library(shiny)
source("R/midi_to_wav.R")

shinyServer(function(input, output) {
  
  observeEvent(input$file, {
     inFile <- input$file
     output_name = paste("output", toString(sample.int(1000,1)), ".wav", sep = "")
     output_path = paste("www/", output_name, sep = "")
     midi_to_wav(inFile$datapath, output = output_path)
     
     output$audio_ui = renderUI(tags$div(id = "audio_player",
                            tags$audio(src = output_name, type = "audio/wav", autoplay = NA, controls = NA)
     ))
     
  })
  
})
