data <- read.csv("target/data.csv", header = TRUE, sep = ",")

Sys.setenv("DISPLAY"=":0")
png("target/output_graph.png")

plot.new()

plot(data[, 1], type = "n", xlim = range(data[, 1]), ylim = range(data[, -1]), xlab = "density", ylab = "values", main = "Performance over density")

for (i in 2:ncol(data)) {
  lines(data[, 1], data[, i], col = rainbow(ncol(data))[i - 1])
}

legend("topright", legend = names(data)[2:ncol(data)], fill = rainbow(ncol(data)), bty = "n", cex = 0.8)

dev.off()

