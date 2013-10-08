#################################################
# store path
data.dir <- '/Users/zoe/Desktop/kaggle/Facial Keypoints Detection/'
train.file <- paste0(data.dir, 'training.csv')
test.file <- paste0(data.dir, 'test.csv')

# read data and create data.frame
d.train <- read.csv(train.file, stringsAsFactors=F)

# sneak peek data
str(d.train)
# head(d.train)
im.train <- d.train$Image
d.train$Image <- NULL
head(d.train)

# convert strings to integers
# install.packages('doMC') for parallel
library(doMC)
registerDoMC()
im.train <- foreach(im = im.train, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}
str(im.train)

# same for test data
d.test <- read.csv(test.file, stringsAsFactors=F)
im.test <- d.test$Image
d.test$Image <- NULL
registerDoMC()
im.test <- foreach(im = im.test, .combine = rbind) %dopar% {
	as.integer(unlist(strsplit(im, " ")))
}
str(im.test)

# save data, done
# save(d.train, im.train, d.test, im.test, file = 'data.Rd')

#################################################
# visualize data

#rev reverse the resulting vector to match the interpretation of
#R'simagefunction (which expects the origin to be in the lower left corner)
# visualize image 4
im<-matrix(data=rev(im.train[4,]), nrow=96,ncol=96)
image(1:96, 1:96, im, col = gray((0:255)/255))

# color coordinates for the eyes and nose
points(96-d.train$nose_tip_x[4],  96-d.train$nose_tip_y[4],  col='red')
points(96-d.train$left_eye_center_x[4], 96-d.train$left_eye_center_y[4], col='blue')
points(96-d.train$right_eye_center_x[4], 96-d.train$right_eye_center_y[4], col='green')

# check how nose points are distributed
for (i in 1:nrow(d.train)) {
	points(96-d.train$nose_tip_x[i], 96-d.train$nose_tip_y[i],  col='red')
}

# see who's the outlier?!
#idx <-which.max(d.train$nose_tip_x)
# omg outlier image DiCaprio's nose is mislabeled
idx<-which.min(d.train$nose_tip_x)
im <-matrix(data=rev(im.train[idx, ]), nrow=96, ncol=96)
image(1:96, 1:96, im, col = gray((0:255)/255))
points(96-d.train$nose_tip_x[idx], 96-d.train$nose_tip_y[idx],  col='red')


#################################################
# simple benchmark
# naively use column mean as keypoints
p<-matrix(data=colMeans(d.train, na.rm=T), nrow=nrow(d.test), ncol=ncol(d.train), byrow=T)
predictions <-data.frame(ImageID = 1:nrow(d.test) , p )
#head(predictions)
#expected submission format has one keypoint per row, but can easily get that with reshape2 library
#install.packages('reshape2')
library(reshape2)
submission <- melt(predictions, id.vars="ImageID", variable.name="FeatureName", value.name="Location")
head(submission)


# join this with the sample submission file to preserve the same order of entries and save the result
example.submission <- read.csv(paste0(data.dir, 'submissionFileFormat.csv'))
sub.col.names <-names(example.submission)
example.submission$Location <- NULL
submission<- merge(example.submission, submission, all.x=T, sort=F)
submission<- submission[ , sub.col.names]
write.csv(submission, file = "submission_means.csv", quote=F, row.names=F)

#################################################
# using image patches

# start with "left_eye_center"
coord <- "left_eye_center"
patch_size <-10
# a square of 21x21 pixel
coord_x <-paste(coord, "x", sep="_")
coord_y <-paste(coord, "y", sep="_")

patches<-foreach(i=1:nrow(d.train), .combine=rbind)%do%{
	im<-matrix(data=im.train[i,], nrow=96, ncol=96)
	x <-d.train[i, coord_x]
	y <-d.train[i, coord_y]
	x1<-(x-patch_size)
	x2<-(x+patch_size)
	y1<-(y-patch_size)
	y2<-(y+patch_size)	
	if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
		as.vector(im[x1:x2, y1:y2])
	else
		NULL
}
mean.patch <-matrix(data=colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)

# see what average eye looks like~~
image(1:21, 1:21, mean.patch[21:1, 21:1], col=gray((0:255)/255))

# use average_patch (how many pixels we are going to move in each direction when searching for the keypoints) 
#to search for the same keypoints in the test images
search_size <-2
mean_x <-mean(d.train[,coord_x], na.rm=T)
mean_y <-mean(d.train[,coord_y], na.rm=T)
x1 <- as.integer(mean_x)-search_size
x2 <- as.integer(mean_x)+search_size
y1 <- as.integer(mean_y)-search_size
y2 <- as.integer(mean_y)-search_size
# use expand.grid to build a data frame with all combinations of x's and y's:
params <-expand.grid(x=x1:x2, y=y1:y2)

# given a test image, try all the combinations, and see which one best matches the average_patch
# do this by taking patches of the test images aroudn these points and measuring their correlation with the average_patch
# do this for all coordinates

# list the coordinates we have to predict
coordinate.names <- gsub("_x", "", names(d.train)[grep("_x", names(d.train))])
 
 
# for each one, compute the average patch
mean.patches <- foreach(coord = coordinate.names) %dopar% {
	cat(sprintf("computing mean patch for %s\n", coord))
	coord_x <- paste(coord, "x", sep="_")
	coord_y <- paste(coord, "y", sep="_")
 
	# compute average patch
	patches <- foreach (i = 1:nrow(d.train), .combine=rbind) %do% {
		im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
		x   <- d.train[i, coord_x]
		y   <- d.train[i, coord_y]
		x1  <- (x-patch_size)
		x2  <- (x+patch_size)
		y1  <- (y-patch_size)
		y2  <- (y+patch_size)
		if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
		{
			as.vector(im[x1:x2, y1:y2])
		}
		else
		{
			NULL
		}
	}
	matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
}


p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind) %dopar% {
	# the coordinates we want to predict
	coord   <- coordinate.names[coord_i]
	coord_x <- paste(coord, "x", sep="_")
	coord_y <- paste(coord, "y", sep="_")
 
	# the average of them in the training set (our starting point)
	mean_x  <- mean(d.train[, coord_x], na.rm=T)
	mean_y  <- mean(d.train[, coord_y], na.rm=T)
 
	# search space: 'search_size' pixels centered on the average coordinates 
	x1 <- as.integer(mean_x)-search_size
	x2 <- as.integer(mean_x)+search_size
	y1 <- as.integer(mean_y)-search_size
	y2 <- as.integer(mean_y)+search_size
 
	# ensure we only consider patches completely inside the image
	x1 <- ifelse(x1-patch_size<1,  patch_size+1,  x1)
	y1 <- ifelse(y1-patch_size<1,  patch_size+1,  y1)
	x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
	y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
 
	# build a list of all positions to be tested
	params <- expand.grid(x = x1:x2, y = y1:y2)
 
	# for each image...
	r <- foreach(i = 1:nrow(d.test), .combine=rbind) %do% {
		if ((coord_i==1)&&((i %% 100)==0)) { cat(sprintf("%d/%d\n", i, nrow(d.test))) }
		im <- matrix(data = im.test[i,], nrow=96, ncol=96)
 
		# ... compute a score for each position ...
		r  <- foreach(j = 1:nrow(params), .combine=rbind) %do% {
			x     <- params$x[j]
			y     <- params$y[j]
			p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
			score <- cor(as.vector(p), as.vector(mean.patches[[coord_i]]))
			score <- ifelse(is.na(score), 0, score)
			data.frame(x, y, score)
		}
 
		# ... and return the best
		best <- r[which.max(r$score), c("x", "y")]
	}
	names(r) <- c(coord_x, coord_y)
	r
}

# prepare file for submission
predictions        <- data.frame(ImageId = 1:nrow(d.test), p)
submission         <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
example.submission <- read.csv(paste0(data.dir, 'submissionFileFormat.csv'))
sub.col.names      <- names(example.submission)
example.submission$Location <- NULL
 
submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]
 
write.csv(submission, file="tutorial.csv", quote=F, row.names=F)





