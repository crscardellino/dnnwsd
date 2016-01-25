plot(0:10, a_err_mean, col="royalblue", type='o', ylim=c(0,1), xlim=c(0,10), main="Error of verb classification", xlab="Iterations", ylab="Root mean squared error")
lines(0:10, b_err_mean, col="darkorange", type='o', pch=22, lty="dashed")
legend("topright", c("Bootstrap Only", "Bootstrap with Active Learning"), col=c("darkorange", "royalblue"), pch=c(22,21), lty=c(2,1))
legend("topright", c("Bootstrap Only", "Bootstrap with Active Learning"), col=c("darkorange", "royalblue"), pch=c(22,21), lty=c(2,1), bty="n")
plot(0:10, a_err_mean, col="royalblue", type='o', ylim=c(0,1), xlim=c(0,10), main="Error of verb classification", xlab="Iterations", ylab="Root mean squared error")
lines(0:10, b_err_mean, col="darkorange", type='o', pch=22, lty="dashed")
legend("topright", c("Bootstrap Only", "Bootstrap with Active Learning"), col=c("darkorange", "royalblue"), pch=c(22,21), lty=c(2,1), bty="n")
plot(0:10, a_err_mean, col="royalblue", type='o', ylim=c(0,1), xlim=c(0,10), main="Error of verb classification", xlab="Iterations", ylab="Root mean squared error", xaxt='n')
lines(0:10, b_err_mean, col="darkorange", type='o', pch=22, lty="dashed")
legend("topright", c("Bootstrap Only", "Bootstrap with Active Learning"), col=c("darkorange", "royalblue"), pch=c(22,21), lty=c(2,1), bty="n")
axis(1, at=0:10)
plot(0:10, a_err_mean, col="royalblue", type='o', ylim=c(0,1), xlim=c(0,10), main="Error of verb classification", xlab="Iterations", ylab="Root mean squared error", xaxt='n')
lines(0:10, b_err_mean, col="darkorange", type='o', pch=22, lty="dashed")
legend("topright", c("Bootstrap Only", "Bootstrap with Active Learning"), col=c("darkorange", "royalblue"), pch=c(22,21), lty=c(2,1), bty="n")
axis(1, at=0:10)
plot(0:10, a_acc_mean, col="royalblue", type='o', ylim=c(0,100), xlim=c(0,10), main="Stratified 10-Fold Cross Validation Accuracy", xlab="Iterations", ylab="Accuracy", xaxt='n')
lines(0:10, b_acc_mean, col="darkorange", type='o', pch=22, lty="dashed")
axis(1, at=0:10)
legend("bottomright", c("Bootstrap Only", "Bootstrap with Active Learning"), col=c("darkorange", "royalblue"), pch=c(22,21), lty=c(2,1), bty="n")
for(name <- names(a_acc)) { cat(name) }
for(name in names(a_acc)) { cat(name) }
for(name in names(a_acc)) { cat(paste(name, "\n")) }
for(name in names(b_acc)) { cat(paste(name, "\n")) }
n1 <- names(b_acc)[1]
n1
b_acc$n1
b_acc[n1]
b_acc["creer"]
dir()
source("plotting.R)
source("plotting.R")
source(plotting.R)
source("plotting.R")
source("plotting.R", echo=T)
plot_verbs(a_acc, b_acc, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright")
a_acc
a_acc["apuntar"]
0:10
b_acc
plot_verbs(a_err, b_err, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright")
debug(plot_verbs)
plot_verbs(a_err, b_err, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright")
plot_verbs(a_acc, b_acc, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright")
n
n
verb
type
n
plot_tile
plot_title
n
plot_title
a[verb]
0:10
n
c
plot_verbs(a_acc, b_acc, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright")
n
n
n
plot(0:10, a[verb], col = "orangered2", type = "o", ylim = c(0, 
    100), xlim = c(0, 10), main = plot_title, xlab = "Iterations", 
plot(0:10, a[verb], col = "orangered2", type = "o")
plot(0:10, a_acc[verb], col = "orangered2", type = "o")
plot(0:10, a_acc["apuntar"], col = "orangered2", type = "o")
plot(0:10, as.vector(a_acc["apuntar"]), col = "orangered2", type = "o")
as.array(a_acc["apuntar"])
as.vector(a_acc["apuntar"])
plot(0:10, as.vector(a_acc$apuntar), col = "orangered2", type = "o")
as.vector(a_acc["apuntar"])
a_acc$apuntar
as.vector(a_acc["apuntar"])
as.array(a_acc["apuntar"])
as.array(a_acc[,"apuntar"])
plot(0:10, a_acc[verb], col = "orangered2", type = "o")
plot(0:10, a_acc["apuntar"], col = "orangered2", type = "o")
plot(0:10, a_acc[,"apuntar"], col = "orangered2", type = "o")
plot(0:10, a_acc[,"apuntar"], col = "orangered2", type = "o")
plot(0:10, a_acc[,"apuntar"], col = "orangered2", type = "o")
plot(0:10, smooth.spline(a_acc[,"apuntar"], df=6), col = "orangered2", type = "o")
plot(smooth.spline(a_acc[,"apuntar"], df=6), col = "orangered2", type = "o")
plot(smooth.spline(a_acc[,"apuntar"], df=5), col = "orangered2", type = "o")
plot(smooth.spline(a_acc[,"apuntar"], df=3), col = "orangered2", type = "o")
plot(smooth.spline(a_acc[,"apuntar"], df=6), col = "orangered2", type = "o")
plot(smooth.spline(a_acc[,"apuntar"], df=5), col = "orangered2", type = "o")
help(smooth.spline)
plot(smooth.spline(a_acc[,"apuntar"], 0:10, df=5), col = "orangered2", type = "o")
plot(smooth.spline(0:10, a_acc[,"apuntar"], df=5), col = "orangered2", type = "o")
source("plotting.R", echo=T)
plot_verbs(a_acc, b_acc, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright", c(0,100))
source("plotting.R", echo=T)
plot_verbs(a_acc, b_acc, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright", c(0,100))
plot_verbs(a_acc, b_acc, "classification.error.comparison", "Error of verb classification", "Root mean squared error", "topright", c(0,1))
plot_verbs(a_err, b_err, "classification.error.comparison", "Error of verb classification", "Root mean squared error", "topright", c(0,1))
plot_verbs(a_acc, b_acc, "classification.crosvalidation.accuracy.comparison", "Stratified 10-Fold Cross-Validation Accuracy", "Accuracy", "bottomright", c(0,100))
ls
apuntar <- read.table("bootstrap/apuntar.fscores.txt", head=F)
apuntar
bapuntar <- read.table("bootstrap/apuntar.fscores.txt", head=F)
bapuntar
aapuntar <- read.table("bootstrap.al/apuntar.fscores.txt", head=F)
aapuntar <- read.table("bootstrap.al/apuntar.fscores.txt", head=F)
aapuntar <- read.table("bootstrap.al/apuntar.fscores.txt", head=F)
aapuntar <- read.table("bootstrap.al/apuntar.fscores.txt", head=F)
aapuntar
bapuntarm <- data.matrix(bapuntar)[:,2:ncol(bapuntar)]
data.matrix(bapuntar)
data.matrix(bapuntar)[:,2:ncol(bapuntar)]
(data.matrix(bapuntar))[:,2:ncol(bapuntar)]
m <- data.matrix(bapuntar)
m[:,2:ncol(m)]
m[,2:ncol(m)]
data.matrix(bapuntar)[,2:ncol(bapuntar)]
data.matrix(bapuntar)[,2:ncol(bapuntar)]
t(data.matrix(bapuntar)[,2:ncol(bapuntar)])
cm.colors(12)
bapuntar
aapuntar
aapuntar
aapuntar$V1
names <- aapuntar$V1
names
id
plot(aapuntar$V2, xlab=names)
plot(aapuntar$V2, names)
plot(aapuntar$V2, aapuntar$V1)
plot(aapuntar$V1, aapuntar$V2)
plot(aapuntar$V2)
plot(aapuntar$V2, ylab=c(1,10))
axis(1, at=1:10)
t(data.matrix(bapuntar)[,2:ncol(bapuntar)])
mb <- t(data.matrix(bapuntar)[,2:ncol(bapuntar)])
mb
plot(mb, type='o', pch=22)
plot(mb, type='o', pch=22)
matplot(mb, type='o', pch=22)
matplot(mb, type='o', pch=22, col=cm.color(2))
matplot(mb, type='o', pch=22, col=cm.colors(2))
matplot(mb, type='o', pch=22, col=terrain.colors(2))
matplot(mb, type='o', pch=22, col=topo.colors(2))
matplot(mb, type='o', pch=22, col=topo.colors(2), ylim=c(0,1))
axis(1, at=1:10)
lengend("bottomleft", legend=bapuntar$V1, col=topo.colors(2), bty="n")
legend("bottomleft", legend=bapuntar$V1, col=topo.colors(2), bty="n")
legend("bottomleft", legend=bapuntar$V1, col=topo.colors(2))
legend("bottomleft", legend=bapuntar$V1, col=topo.colors(2), pch=1)
legend("bottomleft", legend=bapuntar$V1, col=topo.colors(2), pch=22)
matplot(mb, type='o', pch=22, col=topo.colors(2), ylim=c(0,1))
legend("bottomleft", legend=bapuntar$V1, col=topo.colors(2), pch=22)
matplot(mb, type='o', pch=22, col=topo.colors(2), ylim=c(0,1))
legend("bottomleft", legend=bapuntar$V1, col=topo.colors(2), pch=22, bty='n')
bapuntar
matplot(mb, type='o', pch=22, col=topo.colors(2), ylim=c(0,1))
ncol(mb)
nrow(bapuntar)
source("plotting.R")
source("plotting.R")
verbs
source("plotting.R")
plot_verbs(verbs, fscores, "F-Score per sense", 16, c(0,1))
# plot_verbs(verbs, "fscores", "F-Score per sense", 16, c(0,1))
source("plotting.R")
plot_verbs(verbs, "fscores", "F-Score per sense", 16, c(0,1))
btm
source("plotting.R")
plot_verbs(verbs, "fscores", "F-Score per sense", 16, c(0,1))
source("plotting.R")
plot_verbs(verbs, "fscores", "F-Score per sense", 16, c(0,1))
source("plotting.R")
source("plotting.R")
plot_verbs(verbs)
source("plotting.R")
plot_verbs(verbs)
dir()
dir("bootstrap")
library(ggplot2)
install.packages("ggplot2")
1
install.packages("ggplot2")
install.packages("ggplot2")
library(ggplot2)
apuntar <- read.table("bootstrap.al/apuntar.counts.txt", header=F)
apuntar
data.matrix(apuntar)
data.matrix(apuntar)[,2:ncol(apuntar)]
t(data.matrix(apuntar)[,2:ncol(apuntar)])
t(data.matrix(apuntar)[,2:ncol(apuntar)])
data.matrix(apuntar)[,2:ncol(apuntar)]
t(data.matrix(apuntar)[,2:ncol(apuntar)])
data.frame(t(data.matrix(apuntar)[,2:ncol(apuntar)]))
ma <- data.frame(t(data.matrix(apuntar)[,2:ncol(apuntar)]))
ma
names(ma) <- apuntar$V1
ma
ggplot(ma, aes(x=1:10, fill=names(ma))
)
ggplot(ma, aes(x=1:10, fill=names(ma)))
ggplot(ma, aes(x=1:10, y=ma, fill=names(ma)))
ggplot(ma)
matplot(ma)
apuntar
names(apuntar) <- c("", 1,2,3,4,5,6,7,8,9,10)
apuntar
names(apuntar) <- c(,1,2,3,4,5,6,7,8,9,10)
names(apuntar) <- c("",1,2,3,4,5,6,7,8,9,10)
apuntar
ind(apuntar)
names(apuntar) <- c(0,1,2,3,4,5,6,7,8,9,10)
apuntar
rownames(apuntar)
names(apuntar) <- c("ind",1,2,3,4,5,6,7,8,9,10)
apuntar
names(apuntar) <- c("sense",1,2,3,4,5,6,7,8,9,10)
ggplot(apuntar, aes(x=1:10, y=value, fill=sense)) + geom_area(position="fill")
ggplot(apuntar, aes(x=1:10, y=value, fill=sense)) + geom_area(position="fill")
melt
mel(
)
melt()
ggplot(apuntar, aes(x=1:10, y=1:10, fill=sense)) + geom_area(position="fill")
aes
aes(x=1:10)
aes(x=c(1,10))
t <- c(0,0,1,1,2,2,3,3)
aex(x=t)
aes(x=t)
ggplot(apuntar)
apuntar
apuntar
apuntar <- read.table("bootstrap.al/apuntar.stacked.txt", header=F)
apuntar
ggplot(apuntar, aes(x=V1, y=V3, group=V2, fill=V2)) + geom_area(position="fill")
library(ggplot2)
ggplot(apuntar, aes(x=V1, y=V3, group=V2, fill=V2)) + geom_area(position="fill")
apuntar
 aes(x=V1, y=V3, group=V2, fill=V2)
ggplot(apuntar, aes(x=V1, y=V3, group=V2, fill=V2))
ggplot(apuntar, aes(x=V1, y=V3, group=V2, fill=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, group=V2, fill=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, group=V2, fill=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, orderd=V1)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V3)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V2)) + geom_area(position="fill")
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill")
draw.colors = topo.colors(11)
draw.colors
ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") + guides(fill=F)
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a + guides(fill=F)
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
apuntar
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11")
a
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"))
a
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"), values=draw.colors, name="Verb Sense")
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"), name="Verb Senses") 
a
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"), name="Verb Senses") + guide(fill = guide_legend(reverse=T))
a <- ggplot(apuntar, aes(x=V1, y=V3, fill=V2, order=V1)) + geom_area(position="fill") 
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"), name="Verb Senses") + guides(fill = guide_legend(reverse=T))
a
apuntar$V2
length(apuntar$V2)
apuntar$V2[1:length(apuntar$V2)/10]
apuntar$V2[1:11]
size(apuntar$V2)
ncol(apuntar$V2)
nrow(apuntar$V2)
length(apuntar$V2)
apuntar$V2[1:(length(apuntar$V2)/10)]
names(apuntar)
names(apuntar) <- c("Iterations", "Verb Senses", "Count")
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=Verb.Senses)) + geom_area(position="fill") 
a
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=Verb Senses)) + geom_area(position="fill") 
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill="Verb Senses")) + geom_area(position="fill") 
a
names(apuntar) <- c("Iterations", V2, "Count")
names(apuntar) <- c("Iterations", "V2", "Count")
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") 
a
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") + ylim(0,100)
a
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") + scale_y_continuous(formatter='percent')
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") + scale_y_continuous(formatter="percent")
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") + scale_y_continuous(formatter = 'percent')
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") + scale_y_continuous(labels=percent)
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position="fill") + scale_y_continuous(labels='percent')
a
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area()
a
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a
a + scale_x_discrete(breaks=1:10)
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a + scale_x_discrete(breaks=1:10)
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_x_discrete(breaks=1:10)
a <- a + scale_fill_discrete(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"), name="Verb Senses") + guides(fill = guide_legend(reverse=T))
a
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_x_discrete(breaks=1:10)
a <- a + scale_fill_discrete(breaks=apuntar$V2[1:(length(apuntar$V2)/10)], name="Verb Senses") + guides(fill = guide_legend(reverse=T))
a
a <- a + scale_fill_manual(breaks=c("apuntar-4", "apuntar-10", "apuntar-1", "apuntar-2", "apuntar-3", "apuntar-5", "apuntar-6", "apuntar-7", "apuntar-8", "apuntar-9", "apuntar-11"), values=draw.colors, name="Verb Sense")
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_x_discrete(breaks=1:10)
a <- a + scale_fill_manual(breaks=apuntar$V2[1:(length(apuntar$V2)/10)], name="Verb Senses", values=draw.colors) + guides(fill = guide_legend(reverse=T))
a
draw.colors
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_fill_discrete(breaks=apuntar$V2[1:(length(apuntar$V2)/10)], name="Verb Senses") + guides(fill = guide_legend(reverse=T))
a
source("plotting.R")
verbs
plot_counts(verbs)
source("plotting.R")
plot_counts(verbs)
source("plotting.R")
plot_counts(verbs)
source("plotting.R")
plot_counts(verbs)
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
library(ggplot2)
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_fill_discrete(breaks=apuntar$V2[1:(length(apuntar$V2)/10)], name="Verb Senses") + guides(fill = guide_legend(reverse=T))
a
a + scale_fill_manual(breaks=apuntar$V2[1:(length(apuntar$V2)/10)])
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_fill_manual(breaks=apuntar$V2[1:(length(apuntar$V2)/10)], name="Verb Senses") + guides(fill = guide_legend(reverse=T))
a
gg_color_hue <- function(n) {
  hues = seq(15, 375, length=n+1)
  hcl(h=hues, l=65, c=100)[1:n]
}
gg_color_hue(11)
gg_color_hue(10)
gg_color_hue(11)
c?
))
help(c)
help(lapply)
draw.colors = gg_color_hue(11)
apuntar $V1
apuntar$V1
apuntar
apuntar$V2
alf <- apuntar
alf$V2[1:(length(alf$V2)/10)]
verb.sense <- alf$V2[1:(length(alf$V2)/10)]
draw.colors
verb.sense
cbind(verb.sense, draw.colors)
verb.sense.is.vector
is.vector(verb.sense)
is.array(verb.sense)
is.list(verb.sense)
is.data.frame(verb.sense)
is.table(verb.sense)
is.matrix(verb.sense)
is.pairlist
as.list(verb.sense)
c(verb.sense, draw.colors)
help(c)
help(c)
c("apuntar-4"="#64B200")
c("apuntar-4"="#64B200", "apuntar-10"="#00BD5C")
dframe <- c("apuntar-4"="#64B200", "apuntar-10"="#00BD5C")
dframe
is.data.frame(dframe)
typeof(verb.sense)
typeof(draw.colors)
verb.sense
as.data.frame(verb.sense)
typeof(dframe)
length(dframe)
size(dframe)
is.character(dframe)
class(dframe)
class(verb.sense)
class(draw.colors)
c(draw.colors)
class(c(draw.colors))
is.factor(verb.sense)
is.factor(draw.colors)
rbind(draw.colors, verb.sense)
rbind(draw.colors, as.character(verb.sense))
factor(rbind(as.character(verb.sense), draw.colors))
dframe
is.factor(dframe)
c(rbind(as.character(verb.sense), draw.colors))
draw.colors
class(dframe)
as.character(rbind(as.character(verb.sense), draw.colors))
as.character(cbind(as.character(verb.sense), draw.colors))
as.character(c(as.character(verb.sense), draw.colors))
as.c(as.character(verb.sense), draw.colors)
c(as.character(verb.sense), draw.colors)
as.vector(verb.sense)
as.vector(draw.colors)
class(Map)
Map(c, verb.sense, draw.colors)
unlist(Map(c, verb.sense, draw.colors))
unlist(Map(c, as.character(verb.sense), draw.colors))
Map(c, as.character(verb.sense), draw.colors)
help(Map)
help(mapply)
help(mapply)
mapply(c, as.character(verb.sense), draw.colors)
dframe
class(dframe)
as.character(mapply(c, as.character(verb.sense), draw.colors))
as.character(dframe)
dframe$`apuntar-10`
colnames(dframe)
rownames(dframe)
dframe[0]
dframe[1]
dframe[3]
dframe[2]
draw.colors
dframe <- c("apuntar-4"="#F8766D", "apuntar-10"="#DB8E00")
dframe
dframe[0]
dframe[1]
class(dframe[1])
x <- 1:4
names(x) <- letters[1:4]
x
names(dframe)
class(dframe)
dframe$"apuntar-4"
dframe$apuntar-4
dframe$`apuntar-4`
c(x)
class(x)
names(draw.colors) <- verb.sense
draw.colors
a + scale_colour_manual(values=draw.colors)
a <- ggplot(apuntar, aes(x=Iterations, y=Count, fill=V2)) + geom_area(position='fill')
a <- a + scale_fill_manual(breaks=apuntar$V2[1:(length(apuntar$V2)/10)], name="Verb Senses", values=draw.colors) + guides(fill = guide_legend(reverse=T))
a
draw.colors
source("plotting.R")
plot_verbs(verbs)
plot_verbs(verbs)
source("plotting.R")
dir()
source("plotting.R")
source("plotting.R", echo=T)
source("plotting.R", echo=T)
plot_verbs(verbs)
verbs
gg_color_hue(11)
for(verb in verbs) {
    plot_title <- paste("Population percent per sense for \"", verb, "\": Bootstrap Task",  sep="")
    btf <- read.table(paste("bootstrap/", verb, ".stacked.txt", sep=""), header=F)
    verb.senses <- btf$V2[1:(length(btf$V2)/10)]
    draw.colors <- gg_color_hue(length(verbs.senses))
    names(draw.colors) <- verb.senses
    btg <- ggplot(btf, aes(x=V1, y=V3, fill=V2))
    btg <- btg + geom_area(position='fill')
    btg <- btg + scale_x_discrete(breaks=1:10)
    btg <- btg + xlab("Iterations") + ylab("Population Percent")
    btg <- btg + scale_fill_manual(breaks=verb.senses, name="Verb Senses",
    values=draw.colors)
    btg <- btg + guides(fill=guide_legend(reverse=T))
    filename <- paste("plots/counts/", verb, ".counts.bt.pdf", sep="")
    ggsave(filename, plot = btg, width = 7.5, height = 5)
    plot_title <- paste("Population percent per sense for \"", verb, "\": Active Learning Task", sep="")
    alf <- read.table(paste("bootstrap.al/", verb, ".stacked.txt", sep=""),
    header=F)
    verb.senses <- alf$V2[1:(length(alf$V2)/10)]
    draw.colors <- gg_color_hue(length(verbs.senses))
    names(draw.colors) <- verb.senses
    alg <- ggplot(alf, aes(x=V1, y=V3, fill=V2))
    alg <- alg + geom_area(position='fill')
    alg <- alg + scale_x_discrete(breaks=1:10)
    alg <- alg + xlab("Iterations") + ylab("Population Percent")
    alg <- alg + scale_fill_manual(breaks=verb.senses, name="Verb Senses",
    values=draw.colors)
    alg <- alg + guides(fill=guide_legend(reverse=T))
    filename <- paste("plots/counts/", verb, ".counts.al.pdf", sep="")
    ggsave(filename, plot = alg, width = 7.5, height = 5)
  }
source("plotting.R", echo=T)
plot_counts(verbs)
source("plotting.R", echo=T)
plot_counts(verbs)
source("plotting.R")
plot_fscore(verbs)
plot_counts(verbs)
source("plotting.R")
plot_counts(verbs)
plot_fscore(verbs)
source("plotting.R")
plot_counts(verbs)
source("plotting.R")
plot_counts(verbs)
