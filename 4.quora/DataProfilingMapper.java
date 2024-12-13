import java.io.IOException;

import java.util.Map;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;


public class DataProfilingMapper 
        extends Mapper<LongWritable, Text, Text, IntWritable> {

    private static final Set<String> correctQuestionKeys = new HashSet<>(Arrays.asList(
            "qid",
            "url",
            "title",
            "creationTime",
            "followerCount",
            "viewCount",
            "numAnswers",
            "numMachineAnswers",
            "isLocked",
            "isTrendyQuestion",
            "asker",
            "answers"
        ));

    private static final Set<String> correctAnswerKeys = new HashSet<>(Arrays.asList(
        "aid",
        "url",
        "content",
        "author",
        "isSensitive",
        "isShortContent",
        "creationTime",
        "numViews",
        "numUpvotes",
        "numShares",
        "numComments",
        "comments"
    ));

    // Method to recursively count comments, including nested comments
    private static int countComments(JsonArray comments) {
        int num = comments.size();
        for (JsonElement commentElement : comments) {
            JsonObject comment = commentElement.getAsJsonObject();
            if (comment.has("comments")) {
                JsonElement subCommentsElement = comment.get("comments");
                if (subCommentsElement != null && subCommentsElement.isJsonArray()) {
                    num += countComments(subCommentsElement.getAsJsonArray());
                }
            }
        }
        return num;
    }

    // Check whether an int is positive
    private static void checkPositiveIntNum(int num, String tag, Context context) 
        throws IOException, InterruptedException {
        if (num < 0) {
            context.write(new Text(tag), new IntWritable(1));
        } else {
            context.write(new Text(tag), new IntWritable(0));
        }
    }

    // Check whether a long is positive
    private static void checkPositiveLongNum(long num, String tag, Context context) 
        throws IOException, InterruptedException {
        if (num < 0) {
            context.write(new Text(tag), new IntWritable(1));
        } else {
            context.write(new Text(tag), new IntWritable(0));
        }
    }

    private static void checkBoolean(boolean boolValue, String tag, Context context) 
        throws IOException, InterruptedException {
        if (boolValue) {
            context.write(new Text(tag), new IntWritable(1));
        } else {
            context.write(new Text(tag), new IntWritable(0));
        }
    }
    
    
    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        
        // Parse the line to json object
        JsonObject questionJson = JsonParser.parseString(value.toString()).getAsJsonObject();
	    context.write(new Text("Question number"), new IntWritable(1));

        // Check if question key is correct
        if (!questionJson.keySet().equals(correctQuestionKeys)) {
            // question key not match
            context.write(new Text("Question Key Not Match"), new IntWritable(1));
        } else {
            // Question key match
            context.write(new Text("Question Key Not Match"), new IntWritable(0));
            
            // Parse 'url' and process it
            String url = questionJson.get("url").getAsString();
            if (url.isEmpty()) {
                context.write(new Text("Question Null URL"), new IntWritable(1));
            } else {
                context.write(new Text("Question Null URL"), new IntWritable(0));
            }
            // Remove leading and trailing '/', and split by '/'
            String[] urlSplit = url.trim()
                .replaceAll("^/+", "")
                .replaceAll("/+$", "")
                .split("/");

            // Check if title is null
            if (questionJson.get("title").isJsonNull()) {
                context.write(new Text("Question Null Title"), new IntWritable(1));
            } else {
                context.write(new Text("Question Null Title"), new IntWritable(0));
            }

            // Check if the question is locked
            checkBoolean(questionJson.get("isLocked").getAsBoolean(), "Question Locked", context);
            // Check if the question is trendy
            checkBoolean(questionJson.get("isTrendyQuestion").getAsBoolean(), "Question Trendy", context);
            // Check if follower count is negative
            checkPositiveIntNum(questionJson.get("followerCount").getAsInt(), "Question Negative follower Count", context);
            // Check if view count is negative
            checkPositiveIntNum(questionJson.get("viewCount").getAsInt(), "Question Negative View Count", context);       
            // Check if Answer Num is negative
            checkPositiveIntNum(questionJson.get("numAnswers").getAsInt(), "Question Negative Answer Num", context);
            // Check if Machine Answer Num is negative
            checkPositiveIntNum(questionJson.get("numMachineAnswers").getAsInt(), "Question Negative Machine Answer Num", context);
            // Check if Timestamp is negative
            checkPositiveLongNum(questionJson.get("creationTime").getAsLong(), "Question Negative Creation Time", context);


            // Parse 'answers' field
            String answersStr = questionJson.get("answers").getAsString();
            JsonArray answersJson = JsonParser.parseString(answersStr).getAsJsonArray();
            context.write(new Text("Answer Number"), new IntWritable(answersJson.size()));

            // Check if number of answers matches 'numAnswers'
            int numAnswersField = questionJson.get("numAnswers").getAsInt();
            if (answersJson.size() != numAnswersField ||
                (numAnswersField == 0 && !urlSplit[0].equals("unanswered"))) {
                context.write(new Text("Answer Number Not Match"), new IntWritable(1));
            } else {
                context.write(new Text("Answer Number Not Match"), new IntWritable(0));
            }

            // Process each answer
            int answerKeyNotMatch = 0, sensitiveAnswer = 0, shortContentAnswer = 0, commentNumNotMatchAnswer = 0;
            int negativeViewNum = 0, negativeUpvoteNum = 0, negativeShareNum = 0, negativeCommentNum = 0; 
            for (JsonElement answerElement : answersJson) {
                JsonObject answerJson = answerElement.getAsJsonObject();

                // Check if keys match for the answer
                if (!answerJson.keySet().equals(correctAnswerKeys)) {
                    answerKeyNotMatch++;
                }

                // Check if answer is sensitive
                if (answerJson.get("isSensitive").getAsBoolean()) {
                    sensitiveAnswer++;
                }
                // Check if answer has short content
                if (answerJson.get("isShortContent").getAsBoolean()) {
                    shortContentAnswer++;
                }
                
                // Check if count is negative
                if (answerJson.get("numViews").getAsInt() < 0) {
                    negativeViewNum++;
                }
                if (answerJson.get("numUpvotes").getAsInt() < 0) {
                    negativeUpvoteNum++;
                }
                if (answerJson.get("numShares").getAsInt() < 0) {
                    negativeShareNum++;
                }
                if (answerJson.get("numComments").getAsInt() < 0) {
                    negativeCommentNum++;
                }

                // Check if comment num is correct
                JsonArray commentsArray = answerJson.get("comments").getAsJsonArray();
                int actualCommentsCount = countComments(commentsArray);

                if (answerJson.get("numComments").getAsInt() != actualCommentsCount) {
                    commentNumNotMatchAnswer++;
                }
            }
            context.write(new Text("Answer Key Not Match"), new IntWritable(answerKeyNotMatch));
            context.write(new Text("Answer Sensitive"), new IntWritable(sensitiveAnswer));
            context.write(new Text("Answer ShortContect"), new IntWritable(shortContentAnswer));
            context.write(new Text("Answer Comment Num Not Match"), new IntWritable(commentNumNotMatchAnswer));
            context.write(new Text("Answer Negative View Num"), new IntWritable(negativeViewNum));
            context.write(new Text("Answer Negative Upvote Num"), new IntWritable(negativeUpvoteNum));
            context.write(new Text("Answer Negative Share Num"), new IntWritable(negativeShareNum));
            context.write(new Text("Answer Negative Comment Num"), new IntWritable(negativeCommentNum));
        }
    }
}
