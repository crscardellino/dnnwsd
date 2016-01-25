#!/usr/bin/env Rscript

require(ggplot2)

args = commandArgs(trailingOnly=T)

verb_name = args[1]
experiment_name = args[2]
file_name = args[3]
output = args[4]

pdf(output, width=10, height=7)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length=n+1)
  hcl(h=hues, l=65, c=100)[1:n]
}

population_progress <- read.csv(file_name, header=F)
names(population_progress) <- c("iteration", "sense", "count")

number_of_iterations <- population_progress$iteration[nrow(population_progress)]
classes <- population_progress$sense[population_progress$iteration == 0]
number_of_classes <- length(classes)
draw.colors <- gg_color_hue(number_of_classes)
names(draw.colors) <- classes
 
p <- ggplot(population_progress, aes(x=iteration, y=count, fill=sense)) +
    labs(x = "Iteration", y = "Population Percent",
         title = paste("Population progress percent\nfor lemma", verb_name, "and experiment", experiment_name))
p + geom_area(position = "fill") + scale_x_continuous(limit = c(0, number_of_iterations), breaks=0:number_of_iterations) +
    scale_y_continuous(label=function(y){return(y*100)}) +
    scale_fill_manual(breaks=classes, name="Verb Senses", values=draw.colors) +
    guides(fill=guide_legend(reverse=T)) + theme(plot.title = element_text(size=16, face="bold", vjust=2))

dev.off()
