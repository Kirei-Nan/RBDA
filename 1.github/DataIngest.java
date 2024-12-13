import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.InputSampler;
import org.apache.hadoop.mapreduce.lib.partition.TotalOrderPartitioner;

import java.io.IOException;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

public class DataIngest {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Path inputPath = new Path(args[0]);
        Path partitionFile = new Path(args[1] + "_partitions.lst");
        Path outputStage = new Path(args[1] + "_staging");
        Path outputOrder = new Path(args[1]);

        // Configure job to prepare for sampling
        Job sampleJob = Job.getInstance(conf, "DataIngestionSamplingStage");
        sampleJob.setJarByClass(DataIngest.class);

        // Use the mapper implementation with zero reduce tasks
        sampleJob.setMapperClass(IdMapper.class);
        sampleJob.setNumReduceTasks(0);
        sampleJob.setOutputKeyClass(LongWritable.class);
        sampleJob.setOutputValueClass(Text.class);
        TextInputFormat.setInputPaths(sampleJob, inputPath);

        // Set the output format to a sequence file
        sampleJob.setOutputFormatClass(SequenceFileOutputFormat.class);
        SequenceFileOutputFormat.setOutputPath(sampleJob, outputStage);

        // Submit the job and get completion code.
        int code = sampleJob.waitForCompletion(true) ? 0 : 1;
        if (code == 0) {
            Job orderJob = Job.getInstance(conf, "DataIngestionSortingStage");
            orderJob.setJarByClass(DataIngest.class);

            // Here, use the identity mapper to output the key/value pairs in the SequenceFile
            orderJob.setMapperClass(Mapper.class);
            orderJob.setReducerClass(ValueReducer.class);

            // Set the number of reduce tasks to an appropriate number for the amount of data being sorted
            orderJob.setNumReduceTasks(10);

            // Use Hadoop's TotalOrderPartitioner class
            orderJob.setPartitionerClass(TotalOrderPartitioner.class);

            // Set the partition file
            TotalOrderPartitioner.setPartitionFile(orderJob.getConfiguration(), partitionFile);

            orderJob.setOutputKeyClass(LongWritable.class);
            orderJob.setOutputValueClass(Text.class);

            // Set the input to the previous job's output
            orderJob.setInputFormatClass(SequenceFileInputFormat.class);
            SequenceFileInputFormat.setInputPaths(orderJob, outputStage);

            // Set the output path to the command line parameter
            TextOutputFormat.setOutputPath(orderJob, outputOrder);

            // Set the separator to an empty string
            orderJob.getConfiguration().set("mapreduce.output.textoutputformat.separator", "");

            // Use the InputSampler to go through the output of the previous job, sample it, and create the partition file
            InputSampler.writePartitionFile(orderJob, new InputSampler.RandomSampler<LongWritable, Text>(.001, 10000));

            // Submit the job
            code = orderJob.waitForCompletion(true) ? 0 : 2;
        }

        // Clean up the partition file and the staging directory
        FileSystem.get(new Configuration()).delete(partitionFile, false);
        FileSystem.get(new Configuration()).delete(outputStage, true);

        System.exit(code);
    }

    public static class IdMapper extends
            Mapper<LongWritable, Text, LongWritable, Text> {
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            LongWritable outkey = new LongWritable();
            String line = value.toString();

            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Object> jsonLine = objectMapper.readValue(line, Map.class);
            long id = Long.parseLong(jsonLine.get("id").toString());

            outkey.set(id);
            context.write(outkey, value);
        }
    }

    public static class ValueReducer extends
            Reducer<LongWritable, Text, Text, NullWritable> {
        public void reduce(LongWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            for (Text value: values) {
                ObjectMapper objectMapper = new ObjectMapper();
                ObjectNode jsonObject = objectMapper.createObjectNode();

                String line = value.toString();
                Map<String, Object> jsonLine = objectMapper.readValue(line, Map.class);

                String content = jsonLine.get("title").toString() + " " + jsonLine.get("body").toString();
                String text = content.replaceAll("[^\\p{L}]", " ").replaceAll("\\r|\\n", " ").replaceAll("\\r", " ");
                String processed_text;
                if (text.length() > 400) processed_text = text.substring(0, 400);
                else processed_text = text;

                // jsonObject.put("id", jsonLine.get("id").toString());
                // jsonObject.put("time", jsonLine.get("created_at").toString());
                // jsonObject.put("text", processed_text);
                // jsonObject.put("source", "GH Archive");

                // String jsonString = objectMapper.writeValueAsString(jsonObject);
                // Text output = new Text(jsonString);
                // context.write(output, NullWritable.get());

                String output = "";
                output = output + jsonLine.get("id").toString();
                output = output + "\t" + jsonLine.get("created_at").toString();
                output = output + "\t" + processed_text;
                output = output + "\t" + "GH Archive";
                context.write(new Text(output), NullWritable.get());
            }
        }
    }
}
