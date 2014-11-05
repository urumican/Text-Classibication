%% Bernoulli Model by Xin Li (932-252-493); Sumanth .


%% Loading the data 

train_data = load('F:\slides\Machine Learning CS534\HW\imp2\20newsgroup\data\train.data');
test_data = load('F:\slides\Machine Learning CS534\HW\imp2\20newsgroup\data\test.data');
train_label = load('F:\slides\Machine Learning CS534\HW\imp2\20newsgroup\data\train.label');
test_label = load('F:\slides\Machine Learning CS534\HW\imp2\20newsgroup\data\test.label');

%% initialization.

count_in_class = zeros(max(train_label),1); % show the 
n = 1; %used in count in the part 
k = 1;
word_feature = zeros(61188,20);

temp_word_feature = zeros(61188,20);
class = 1;
P_i_map = zeros(61188,20);
One = ones(1,20);

%% Preprocess data.

for i = 1:max(train_label) % calculate total number of articles in each class.
    
    count_in_class(i) = size(find(train_label==i),1);
    
end


for class = 1:20 % count the number of the words appeared in each class
    
    article_temp = find(train_label == class);
    
 for trainID = 1:size(train_data,1)
     
     if (max(article_temp) >= train_data(trainID,1)) && (train_data(trainID,1) >= min(article_temp))
     %if (train_data(trainID,1) >= n) && (train_data(trainID,1) <= (n + count_in_class(class) - 1))
           
         word_feature(train_data(trainID,2),class) = word_feature(train_data(trainID,2),class) + 1;
         
     end
          
 end
 
    %n = n + count_in_class(class);
    
end

%N_words = sum(word_feature,2); % calculate the total number that each word appear for each class.

%% Generate parameters from training data.

P_ys = count_in_class./size(train_label,1); % calculate the probability of P(y=k).

for words = 1:61188 % calculate the probability of each word that appear for each class using Laplace Smoothing.
    
    P_i_map(words,:) = (word_feature(words,:) + One) ./ (count_in_class' + 2 * One);
    
end

%% Make predictions for each article.

article_feature = zeros(61188,1);
P = zeros(7505,20);
One1 = ones(size(P_i_map,1),1);
One2 = ones(size(article_feature,1),1);

for articleID = 1:7505
    
    word_temp = find(test_data(:,1) == articleID);
    article_feature(test_data(word_temp,2)) = 1;
    
    for classID = 1:20
        
%         for wordID = 1:61188
%             
%             P(articleID,classID) = P(articleID,classID) + article_feature(wordID) * log(P_i_map(wordID,classID)) + (1-article_feature(wordID)) * log(1-P_i_map(wordID,classID));
%         end
%         
%         P(articleID,classID) = P(articleID,classID) + log(P_ys(classID));
        
         temp = sum(log(P_i_map(:,classID).^(article_feature) .* (One1 - P_i_map(:,classID)).^(One1 - article_feature)));
         

         P(articleID,classID) = temp + log(P_ys(classID));
        
    end
    
    article_feature = zeros(61188,1);

end

% Find the class to which the article belongs.

belong = zeros(size(P,1),1);

for article_ID = 1:size(P,1)
    
    belong(article_ID,1) = find(P(article_ID,:) == max(P(article_ID,:)));
    
end
