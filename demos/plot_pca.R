library(ggplot2)
set.seed(123)

np <- import("numpy")
feature <- np$load("./feature.npy")
df <- read.csv("./total.csv")

my_colormap <- c("P20" = "#F6511D", "P40" = "#FFB400", "P60" = "#00A6ED", "P90" = "#7FB800", "P120" = "#0D2C54")

res.pca <- prcomp(feature,  scale = TRUE)

pca.1 <- res.pca$x[, 1]
pca.2 <- res.pca$x[, 2]
pca_df <- data.frame(V1=pca.1, V2=pca.2)

df <- cbind(pca_df, df)
df$gt <- factor(df$gt, levels=c("P20", "P40", "P60", "P90", "P120"))

x_limits <- c(min(df$V1), max(max(df$V1)))
y_limits <- c(min(df$V2), max(max(df$V2)))

test_df <- subset(df, type == "test")

plot <- ggplot(df, aes(x = V1, y = V2, color=gt)) + 
  coord_cartesian(xlim=x_limits, ylim=y_limits) +
  geom_polygon(data = test_df, stat = "ellipse", aes(fill = gt), alpha = 0.3) +
  geom_point(data=df, size=0.3, alpha=0.8) +
  scale_color_manual(values=my_colormap) +
  scale_fill_manual(values=my_colormap) +
  ggtitle("UMAP Visualization for all data")
ggsave("pca_gt_all.png", plot, width = 4, height = 4)


plot <- ggplot(df, aes(x = V1, y = V2, color=gt)) + 
  coord_cartesian(xlim=x_limits, ylim=y_limits) +
  geom_polygon(data = test_df, stat = "ellipse", aes(fill = gt), alpha = 0.3) +
  geom_point(data=test_df, size=0.3, alpha=0.5) + 
  scale_color_manual(values=my_colormap) +
  scale_fill_manual(values=my_colormap) +
  ggtitle("UMAP Visualization for test set")
ggsave("pca_gt_test.png", plot, width = 4, height = 4)

grafted_df <- subset(df, type == "graft")
plot <- ggplot(grafted_df, aes(x = V1, y = V2, color=gt)) + 
  coord_cartesian(xlim=x_limits, ylim=y_limits) +
  geom_polygon(data = test_df, stat = "ellipse", aes(fill = gt), alpha = 0.3) +
  geom_point(data=grafted_df, size=0.3, alpha=0.5) + 
  scale_color_manual(values=my_colormap) +
  scale_fill_manual(values=my_colormap) +
  ggtitle("UMAP Visualization for graft data")
ggsave("pca_gt_graft.png", plot, width = 4, height = 4)
