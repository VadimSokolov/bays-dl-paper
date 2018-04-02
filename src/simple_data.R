require(tree)
fig_path = "~/papers/dl-stats/paper/fig/"

d = read.csv("../data/simple_data.csv")
t = tree(y ~. , data = d)
summary(t)
plot(t)
text(t, pretty=1)
t
par(mar=c(1,1,1,1))
plot(d$x1, d$x2, ann=FALSE, type='p', pch=16, cex=5, col = as.numeric((d$y))+2, xaxt="n", yaxt="n")
partition.tree(t, label="y", add=TRUE, lwd=20, cex=3)
utils$save_pdf(fig_path, "simple_data_tree")

d = read.csv("../data/circle_data.csv")
t = tree(y ~. , data = d)
summary(t)
plot(t)
text(t, pretty=1)
t
plot(d$x1, d$x2, ann=FALSE, xaxt="n", yaxt="n", type='p', pch=16, cex=5, col = as.numeric((d$y))+2, lwd=10)
partition.tree(t, label="y", add=TRUE, lwd=10, cex=2)
utils$save_pdf(fig_path, "circle_data_tree")


d = read.csv("../data/spiral_data.csv")
t = tree(y ~. , data = d)
summary(t)
plot(t)
text(t, pretty=1)
  t
plot(d$x1, d$x2, ann=FALSE, xaxt="n", yaxt="n", type='p', pch=16, cex=5, col = as.numeric((d$y))+2)
partition.tree(t, label="y", add=TRUE, lwd=24, cex=2)
utils$save_pdf(fig_path, "spiral_data_tree")

