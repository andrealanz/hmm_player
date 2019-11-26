library(shiny)

#solution influenced by https://github.com/XD-DENG/Reactively-Play-Audio-Shiny.git

shinyUI(
  fluidPage(
    titlePanel("HMM Player"),
    
    sidebarLayout(
      
      sidebarPanel(  
        fileInput("file", "Upload MIDI", multiple = FALSE, accept = c("mid", "midi"))
      ),
      
      mainPanel( 
        uiOutput("audio_ui")
      )
      
    )
  )
)



