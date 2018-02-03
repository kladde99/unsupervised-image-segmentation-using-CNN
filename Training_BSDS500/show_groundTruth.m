% show the groundtruth of the first image of the dataset
subplot(1,6,1);imagesc(imdb3.images{1, 1}  );axis image off ; 
  title('OriginalImage');
 subplot(1,6,2);imagesc(groundTruth{1, 1}.Segmentation);axis image off ;
 title('GroundTruth1');
 subplot(1,6,3);imagesc(groundTruth{1, 2}.Segmentation);axis image off ; 
 title('GroundTruth2');
 subplot(1,6,4);imagesc(groundTruth{1, 3}.Segmentation);axis image off ; 
 title('GroundTruth3');
 subplot(1,6,5);imagesc(groundTruth{1, 4}.Segmentation);axis image off ; 
 title('GroundTruth4');
 subplot(1,6,6);imagesc(groundTruth{1, 5}.Segmentation);axis image off ; 
 title('GroundTruth5');