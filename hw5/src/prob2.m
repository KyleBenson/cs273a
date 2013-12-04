problem = 'a'
nclusters = 40

% Read in vocabulary and data (word counts per document)
[vocab] = textread('data/text/vocab.txt','%s');
[did,wid,cnt] = textread('data/text/docword.txt','%d%d%d','headerlines',3);

X=sparse(did,wid,cnt); % reverse with full(X)

D=max(did); %number of docs       
W=max(wid); %size of vocab        
N=sum(cnt); %total number of words

% It is often helpful to normalize by the document length:
Xn = X./repmat(sum(X,2),[1,W]) ; % divide word counts by doc length

if problem == 'a'

[assign, clusters, sumd] = kmeans(Xn, nclusters);

for i = 1:1:nclusters
    fprintf('Cluster %d has %d articles \\\\\n', i, sum(assign==i));
    % print 10 most common words from each cluster
    [sortedWords, indices] = sort(clusters(i,:), 'descend');
    fprintf('Most common words: ');
    for j = 1:1:10
        word = vocab{indices(j)};
        fprintf('%s, ', word);
    end;
    fprintf('\\\\\n');
end;

% print first few lines from each article in the clusters
% containing documents 1, 15, and 30
clustersToPrint = [assign(1) assign(15) assign(30)];
for clusp = clustersToPrint
    fprintf('Text from docs in cluster %d:\n', clusp);
    docsToPrint = (1:1:size(assign,1))';
    docsToPrint = docsToPrint(assign==clusp);
    % don't print more than a dozen documents
    docsToPrint = docsToPrint(1:min(12, size(docsToPrint,1)),:);
    for doc = docsToPrint'
        % view snippet of article itself
        fname = sprintf('data/text/example1/20000101.%04d.txt',doc);
        txt = textread(fname,'%s',10,'whitespace','\r\n');
        fprintf('%s\n',txt{:});
    end;
    fprintf('\n');
    fprintf('\n');
end;

sum_squared_distances = sumd


elseif problem == 'b'

% output most common words in article based on features
[sorted,order] = sort(Xn(i,:), 2, 'descend');
fprintf('Doc %d: ',i); fprintf('%s ',vocab{order(1:10)}); fprintf('\n');


end;