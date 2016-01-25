#!/usr/bin/env Rscript

require(ggplot2)

args = commandArgs(trailingOnly=T)

verb_name = args[1]
file_name = args[2]
output = args[3]

pdf(output, width=10, height=7)

tuning_accuracy <- read.csv(file_name)
tuning_accuracy$iteration <- 1:nrow(tuning_accuracy)

p <- ggplot(tuning_accuracy, aes(iteration)) +
    labs(x = "Iteration", y = "Accuracy",
         title = paste("Accuracy progress for lemma", verb_name))
p + geom_line(aes(y=bow, color="Bag of Words")) +
    geom_line(aes(y=wordvec, color="Word Vectors")) +
    geom_line(aes(y=wordvecpos, color="Word Vectors with PoS")) +
    scale_colour_discrete(name='') +
    coord_cartesian(xlim=c(1,nrow(tuning_accuracy)), ylim=c(0,1)) +
    theme(plot.title = element_text(size=16, face="bold", vjust=2))

dev.off()
