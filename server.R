library(shiny)


shinyServer(function(input, output) {
  
  observeEvent(input$play, {
    insertUI(selector = "#play",
             where = "afterEnd",
             ui = tags$audio(src = "song.mp3", type = "audio/mp3", autoplay = NA, controls = NA)  
             )
  })
})
