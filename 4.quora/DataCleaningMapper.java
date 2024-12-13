import java.io.IOException;

import java.util.Map;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;

import java.text.SimpleDateFormat;
import java.util.Date;

public class DataCleaningMapper 
        extends Mapper<LongWritable, Text, NullWritable, Text> {

    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        
        // Parse the line to json object
        JsonObject questionJson = JsonParser.parseString(value.toString()).getAsJsonObject();

        // Get creation time and question text
        Set<String> questionJsonKeySet = questionJson.keySet();
        if (questionJsonKeySet.contains("url") && questionJsonKeySet.contains("creationTime")) {
            // Parse 'url' and process it
            String url = questionJson.get("url").getAsString();
            long creationTime = questionJson.get("creationTime").getAsLong();

            if (!url.isEmpty() && creationTime >= 0) {
                // Remove leading and trailing '/', and remove "unanswered/" prefix
                String questionString = url.trim()
                    .replaceAll("^/+", "")
		    .replaceAll("/+$", "")
                    .replaceAll("^unanswered/", "")
                    .replaceAll("-|\t", " ")
                    .toLowerCase();

                // Transform date
                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                String dataString = sdf.format(new Date(creationTime /1000)); 

                // Output result using null as key
                context.write(NullWritable.get(), new Text(dataString + "\t" + questionString));
            }
        }
    }
}
