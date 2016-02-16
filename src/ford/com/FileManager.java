package ford.com;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.StringTokenizer;
import java.util.TreeSet;
import java.util.Vector;

import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stemmers.SnowballStemmer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.types.InstanceList;

public class FileManager {
	
	String operationOn = "";
	
	public class LIWCPair{
		public String feature;
		public String featureValue;
		
		// default constructor
		public LIWCPair() {
		}
		
		// customs constructor
		public LIWCPair(String featureData, String featureValueData){
			feature = featureData;
			featureValue = featureValueData;
		}
	}
	
	// create new directory for data folder
	public void createDataFolder(Vector<Entry> data, String folder) throws IOException {
		File mainFolder = new File(folder);
		if (mainFolder.exists()) {
			removeDirectory(mainFolder);
			mainFolder.mkdir();
		}
		else
			mainFolder.mkdir();
		
		// create the folder
		for (int i = 0; i < data.size(); ++i) {
			File classFolder = new File(folder + "/" + data.get(i).label);
			if (!classFolder.exists())
			{
				if(classFolder.mkdirs())
				{
					//System.out.println("Multiple directories are created!");
				}
			}
			
			PrintWriter pw = new PrintWriter(folder + "/" + data.get(i).label + "/" + i + ".txt", "UTF-8" );
			pw.print(data.get(i).text);
			pw.close();
		}
	}
	
	// remove the corresponding directory
	public boolean removeDirectory(File directory) {
		
		if (directory == null)
			return false;
		if (!directory.exists())
			return true;
		if (!directory.isDirectory())
			return false;

		String[] list = directory.list();

		if (list != null) {
			for (int i = 0; i < list.length; i++) {
				File entry = new File(directory, list[i]);

				if (entry.isDirectory())
				{
					if (!removeDirectory(entry))
						return false;
				}
				else
				{
					if (!entry.delete())
						return false;
				}
			}
		}
		
		return directory.delete();
	}
	
	// read raw data provided by the user
	public Vector<Entry> readRawData(String dataFile) throws Exception {
		
		Vector<Entry> out = new Vector<Entry>();
		BufferedReader br = new BufferedReader(new FileReader(dataFile));			
		String line;
		while ((line = br.readLine()) != null) {
		    // process the line.
			String label = line.substring(0,1).trim();
			Entry entry = new Entry();
			entry.label = label;
			entry.text = line.substring(2).trim();
			out.add(entry);
		}
		
		br.close();	
		
		return out;
	}	
	
	public String getFreshAttribute(String term){
		term = term.replace(".", "");
		term = term.replace(",", "");
		term = term.replace(")", "");
		term = term.replace("(", "");
		term = term.replace("]", "");
		term = term.replace("[", "");
		term = term.replace("\"", "");
		term = term.replace("n't", "n");
		term = term.toLowerCase();
		return term;
	}
	
	// stemming
	public static String stem(String string) throws IOException {
		
		SnowballStemmer stemmer = new SnowballStemmer();
		stemmer.setStemmer("english");
		
		StringTokenizer st = new StringTokenizer(string, "\t "); 

	    StringBuilder stringBuilder = new StringBuilder();

	    while(st.hasMoreTokens()) { 
	        
	        	        
	        String term = st.nextToken();
	        term = term.replace(".", "");
			term = term.replace(",", "");
			term = term.replace(")", "");
			term = term.replace("(", "");
			term = term.replace("]", "");
			term = term.replace("[", "");
			term = term.replace("\"", "");
			term = term.toLowerCase();
			
			if(term.contains("?")){
				term = term.replace("?", "");
				stringBuilder.append(" ?");
			}else if(term.contains("!")){
				term = term.replace("!", "");
				stringBuilder.append(" !");
			}else{
				if(stringBuilder.length() > 0 ) {
		            stringBuilder.append(" ");
		        }
				stringBuilder.append(term);			
			}
	    }

	    return stringBuilder.toString();
	}
	
	public void readStopword() throws IOException {
	    BufferedReader br = new BufferedReader(new FileReader("liwc/stopwords.txt"));
	    PrintWriter attWriter = new PrintWriter("liwc/stopword.txt", "UTF-8");
	    
	    try {
	        String line = br.readLine();

	        while (line != null) {
	        	attWriter.write("add(\"" + line + "\");\n");   
	        	line = br.readLine();
	        }
	        
	        br.close();
	        attWriter.close();
	    } catch(Exception e) {
	        br.close();
	        attWriter.close();
	    }
	}
	
	// set container with actual data by maintaining order
	public void setDocuments(Vector<String> documents, Vector<Integer> docIDs) throws IOException{
		File classFolder = new File("train");
		File []classFiles = classFolder.listFiles();
		for (int i = 0; i < classFiles.length; ++i) {			
			File []classTextFiles = classFiles[i].listFiles();						
			String line;			
			for (int j = 0; j < classTextFiles.length; ++j) {				
				BufferedReader br = new BufferedReader(new FileReader(classTextFiles[j]));
				line = br.readLine();
				if (line != null) {
					if(line.length() > 0){
						docIDs.add(Integer.parseInt(classTextFiles[j].getName().replace(".txt", "")));
						documents.add(line.trim());
					}
				}
				else
				{
					System.out.println(classTextFiles[j]);
				}
				
				br.close();
			}
		}		
	}
	
	// set container with actual data by maintaining order
	public void setDocumentsWithNLPFeatures(Vector<String> documents, Vector<Integer> docIDs) throws IOException{
		File classFolder = new File("train");
		File []classFiles = classFolder.listFiles();
		for (int i = 0; i < classFiles.length; ++i) {			
			File []classTextFiles = classFiles[i].listFiles();						
			String line;			
			for (int j = 0; j < classTextFiles.length; ++j) {				
				BufferedReader br = new BufferedReader(new FileReader(classTextFiles[j]));
				line = br.readLine();
				if (line != null) {
					if(line.length() > 0){
						docIDs.add(Integer.parseInt(classTextFiles[j].getName().replace(".txt", "")));
						documents.add(line.trim());
					}
				}
				else
				{
					System.out.println(classTextFiles[j]);
				}
				
				br.close();
			}
		}		
	}
	
	// read auto review data
	public void writeAutoReviewHierarchy(String fileName) throws IOException {

		int lineCounter = 0;
		PrintWriter prepurchaseWriter = new PrintWriter("hierarchicaldata/prepurchase.txt", "UTF-8");
		PrintWriter postpurchaseWriter = new PrintWriter("hierarchicaldata/postpurchase.txt", "UTF-8");
		PrintWriter feedbackWriter = new PrintWriter("hierarchicaldata/feedback.txt", "UTF-8");
		PrintWriter notFeedbackWriter = new PrintWriter("hierarchicaldata/nofeedback.txt", "UTF-8");
		PrintWriter subjectiveWriter = new PrintWriter("hierarchicaldata/subjective.txt", "UTF-8");
		PrintWriter performWriter = new PrintWriter("hierarchicaldata/performance.txt", "UTF-8");
		PrintWriter expectWriter = new PrintWriter("hierarchicaldata/expectation.txt", "UTF-8");
		PrintWriter notexpectWriter = new PrintWriter("hierarchicaldata/notexpectation.txt", "UTF-8");
		PrintWriter posWriter = new PrintWriter("hierarchicaldata/positive.txt", "UTF-8");
		PrintWriter negWriter = new PrintWriter("hierarchicaldata/negative.txt", "UTF-8");
	    BufferedReader br = new BufferedReader(new FileReader(fileName));
	    
	    try {
	    	
	        String line = br.readLine();

	        while (line != null) {
	        	
	        	// pre and post purchase
	        	int lastPositionOfReview = -1;
	        	boolean IsvalidData = false;
        		lineCounter++;
	        	lastPositionOfReview = line.lastIndexOf("pre_purchase");
	        	if(lastPositionOfReview > -1){
	        		IsvalidData = true;
	        		if((line.substring(0, lastPositionOfReview-1).split(" ").length) > 0){
	        			prepurchaseWriter.println(lineCounter + "\t" + (line.substring(0, lastPositionOfReview-1).trim()));
	        		}
	        		
	        		if(!line.contains("Not_expectation")){
	        			if(line.substring(0, lastPositionOfReview-1).trim().length() > 0)
	        				expectWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
	        		}
	        		else{
	        			if(line.substring(0, lastPositionOfReview-1).trim().length() > 0)
	        				notexpectWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
	        		}
	        	}
	        	else{
	        		lastPositionOfReview = line.lastIndexOf("post_purchase");
		        	if(lastPositionOfReview > -1){
		        		IsvalidData = true;
		        		if((line.substring(0, lastPositionOfReview-1).split(" ").length) > 0){
	        				postpurchaseWriter.println(lineCounter + "\t " + (line.substring(0, lastPositionOfReview-1).trim()));
	        			}
		        		
		        		if(!line.contains("Not_performance")){
		        			if(line.substring(0, lastPositionOfReview-1).trim().length() > 0)
		        				performWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
		        		}
		        		else{
		        			if(!line.contains("Not_feedback")){
			        			if(line.substring(0, lastPositionOfReview-1).trim().length() > 0)
			        				feedbackWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
			        		}
		        			else{
		        				notFeedbackWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
		        			}
		        		}
		        	}
	        	}
	        	
	        	// positive and negative
	        	if(IsvalidData){
	        		if(line.contains("Negative")){
	        			if(line.substring(0, lastPositionOfReview-1).trim().length() > 0)
	        				negWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
	        		}
	        		else if(line.contains("positive")){
	        			if(line.substring(0, lastPositionOfReview-1).trim().length() > 0)
	        				posWriter.println(lineCounter + "\t" + line.substring(0, lastPositionOfReview-1).trim());
		        	}
	        	}	    
	        	
	            line = br.readLine();	            
	        }
	        
	    } catch(Exception e) {
	        
	    }
	    finally{
	    	br.close();
	        prepurchaseWriter.close();
	        postpurchaseWriter.close();
	        feedbackWriter.close();
	        expectWriter.close();
	        performWriter.close();
	        notFeedbackWriter.close();
	        posWriter.close();
	        negWriter.close();
	        notexpectWriter.close();
	        subjectiveWriter.close();
	    }
	}
	
	// read auto review data
	public void writeImportantAutoReview(String fileName) throws IOException {

		PrintWriter purchaseWriter = new PrintWriter("hierarchicaldata/purchase.txt", "UTF-8");
		PrintWriter feedbackWriter = new PrintWriter("hierarchicaldata/feedback.txt", "UTF-8");
		PrintWriter subjectiveWriter = new PrintWriter("hierarchicaldata/subjective.txt", "UTF-8");
	    BufferedReader br = new BufferedReader(new FileReader(fileName));
	    
	    try {
	    	
	        String line = br.readLine();

	        while (line != null) {
	        	
	        	// pre and post purchase
	        	int lastPositionOfReview = -1;
	        	boolean IsvalidData = false;
	        	lastPositionOfReview = line.lastIndexOf("pre_purchase");
	        	if(lastPositionOfReview > -1){
	        		IsvalidData = true;
	        		if((line.substring(0, lastPositionOfReview-1).split(" ").length) > 0){
	        			//if(!line.contains("confusing_C"))
	        				purchaseWriter.println("1\t" + (line.substring(0, lastPositionOfReview-1).trim()));
	        		}	        			
	        	}
	        	else{
	        		lastPositionOfReview = line.lastIndexOf("post_purchase");
		        	if(lastPositionOfReview > -1){
		        		IsvalidData = true;
		        		if(line.contains("Not_expectation")){
		        			if((line.substring(0, lastPositionOfReview-1).split(" ").length) > 0){
		        				//if(!line.contains("confusing_C"))
		        					purchaseWriter.println("2\t" + (line.substring(0, lastPositionOfReview-1).trim()));
		        			}
		        		}
		        		else{
		        			if((line.substring(0, lastPositionOfReview-1).split(" ").length) > 0){
		        				//if(!line.contains("confusing_C"))
		        					purchaseWriter.println("2\t " + (line.substring(0, lastPositionOfReview-1).trim()));
		        			}
		        				
		        		}
		        	}
	        	}
	        	        	
	        	// feedback and not_feedback
	        	if(IsvalidData){
	        		if(line.contains("Not_feedback") && line.contains("Not_performance")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				feedbackWriter.println("2\t" + stem(line.substring(0, lastPositionOfReview-1).trim()));
	        		}
	        		else if(line.contains("Not_feedback") && line.contains("performance")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				feedbackWriter.println("1\t" + stem(line.substring(0, lastPositionOfReview-1).trim()));
		        	}
	        	}	
	        	
	        	// positive and negative
	        	if(IsvalidData){
	        		if(line.contains("Negative")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				subjectiveWriter.println("2\t" + stem(line.substring(0, lastPositionOfReview-1).trim()));
	        		}
	        		else if(line.contains("positive")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				subjectiveWriter.println("1\t" + stem(line.substring(0, lastPositionOfReview-1).trim()));
		        	}
	        	}	    
	        	
	            line = br.readLine();	            
	        }
	        
	        br.close();
	        purchaseWriter.close();
	        feedbackWriter.close();
	        subjectiveWriter.close();
	        
	    } catch(Exception e) {
	        br.close();
	        purchaseWriter.close();
	        feedbackWriter.close();
	        subjectiveWriter.close();
	    }
	}
	
	public void filterConfusingReviews() throws IOException {
	    BufferedReader br = new BufferedReader(new FileReader("purchase.txt"));
	    PrintWriter attWriter = new PrintWriter("data/purchase.txt", "UTF-8");
	    
	    try {
	        String line = br.readLine();

	        while (line != null) {
	        	if(!line.split(" ")[0].trim().contains("0"))
	        		attWriter.println(line);   
	        	line = br.readLine();
	        }
	        
	        br.close();
	        attWriter.close();
	    } catch(Exception e) {
	        br.close();
	        attWriter.close();
	    }
	}
	
	public void filterPurchaseTag() throws IOException {
	    BufferedReader br = new BufferedReader(new FileReader("nlpdata/purchase_tag.txt"));
	    PrintWriter attWriter = new PrintWriter("nlpdata/purchase_tags.txt", "UTF-8");
	    
	    try {
	        String line = br.readLine();

	        while (line != null) {
	        	if(line.length() > 0)
	        		attWriter.println(line);   
	        	line = br.readLine();
	        }
	        
	        br.close();
	        attWriter.close();
	    } catch(Exception e) {
	        br.close();
	        attWriter.close();
	    }
	}
	
	public void filterTwoReviews() throws IOException {
	    BufferedReader purchaseReader = new BufferedReader(new FileReader("data/purchase.txt"));
	    BufferedReader tagReader = new BufferedReader(new FileReader("nlpdata/purchase_tagonly.txt"));
	    PrintWriter attWriter1 = new PrintWriter("nlpdata/purchase_split1.txt", "UTF-8");
	    PrintWriter attWriterTag1 = new PrintWriter("nlpdata/purchase_tagonly_split1.txt", "UTF-8");
	    PrintWriter attWriter2 = new PrintWriter("nlpdata/purchase_split2.txt", "UTF-8");
	    PrintWriter attWriterTag2 = new PrintWriter("nlpdata/purchase_tagonly_split2.txt", "UTF-8");
	    
	    try {
	        String line = tagReader.readLine();
	        String sample = purchaseReader.readLine();
	        
	        while (line != null) {
	        	if(line.split(" ")[0].trim().contains("1")){
	        		attWriter1.println(sample);
	        		attWriterTag1.println(line);
	        	}
	        	else{
	        		attWriter2.println(sample);
	        		attWriterTag2.println(line);
	        	}
	        		
	        	line = tagReader.readLine();
	        	sample = purchaseReader.readLine();
	        }
	        
	        tagReader.close();
	        purchaseReader.close();
	        attWriter1.close();
	        attWriterTag1.close();
	        attWriter2.close();
	        attWriterTag2.close();
	    } catch(Exception e) {
	    	tagReader.close();
	    	purchaseReader.close();
	    	purchaseReader.close();
	        attWriter1.close();
	        attWriterTag1.close();
	        attWriter2.close();
	        attWriterTag2.close();
	    }
	}
	
	public void countAllReviews() throws IOException {
	    BufferedReader br = new BufferedReader(new FileReader("msn_data.txt"));
	    PrintWriter attWriter = new PrintWriter("data/purchase.txt", "UTF-8");
	    HashMap<String, Integer> data = new HashMap<String, Integer>();
	    
	    try {
	        String line = br.readLine();

	        while (line != null) {
	        	String []words = line.split("\t");
	        	String key = words[0].trim();
	        	Integer val = 1;
	        	if(data.containsKey(key))
	        		val = data.get(key) + 1;
	        	data.put(key, val);
	        	if(key.equalsIgnoreCase("1")||key.equalsIgnoreCase("2"))
	        		attWriter.println(stem(line));
	        	line = br.readLine();
	        }
	        
	        br.close();
	        attWriter.close();
	    } catch(Exception e) {
	        br.close();
	        attWriter.close();
	    }
	    
	    System.out.println(data);
	}

	//  get sentences from a file
	public void writeAllSentences() throws Exception{		 
		// always start with a model, a model is learned from training data
		InputStream is = new FileInputStream("binFiles/en-sent.bin");
		SentenceModel model = new SentenceModel(is);
		SentenceDetectorME sdetector = new SentenceDetectorME(model);
		
		BufferedReader br = new BufferedReader(new FileReader("backup/edmunds.txt"));
	    PrintWriter attWriter = new PrintWriter("backup/splitedmunds.txt", "UTF-8");
	    
	    try {
	        String line = br.readLine();
	        while (line != null) {	        	
	        	String paragraph = line.trim();
	        	String sentences[] = sdetector.sentDetect(paragraph);
	        	for (int i=0; i < sentences.length; i++){
	        		attWriter.println("0 " + sentences[i]);
	        	}
	        	line = br.readLine();
	        }
	        
	        br.close();
	        is.close();
	        attWriter.close();
	    } catch(Exception e) {
	        br.close();
	        is.close();
	        attWriter.close();
	    }
	}
	
	// create gold standard
	public void readAutoReviewGoldSTD3(String fileName) throws IOException {
		PrintWriter subjectiveWriter = new PrintWriter("hierarchicaldata/objective.txt", "UTF-8");
	    BufferedReader br = new BufferedReader(new FileReader(fileName));
	    
	    try {
	    	
	        String line = br.readLine();

	        while (line != null) {
	        	
	        	// pre and post purchase
	        	int lastPositionOfReview = -1;
	        	boolean IsvalidData = false;
	        	lastPositionOfReview = line.lastIndexOf("pre_purchase");
	        	if(lastPositionOfReview > -1){
	        		IsvalidData = true;        			
	        	}
	        	else{
	        		lastPositionOfReview = line.lastIndexOf("post_purchase");
		        	if(lastPositionOfReview > -1){
		        		IsvalidData = true;
		        	}
	        	}
	        	
	        	
	        	// positive and negative
	        	if(IsvalidData){
	        		if(line.contains("objective")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				subjectiveWriter.println("3 " + stem(line.substring(0, lastPositionOfReview-1).trim()));
	        		}
	        		else if(line.contains("Negative")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				subjectiveWriter.println("2 " + stem(line.substring(0, lastPositionOfReview-1).trim()));
	        		}
	        		else if(line.contains("positive")){
	        			if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0)
	        				subjectiveWriter.println("1 " + stem(line.substring(0, lastPositionOfReview-1).trim()));
		        	}
	        	}	    
	        	
	            line = br.readLine();	            
	        }
	        
	        br.close();
	        subjectiveWriter.close();
	        
	    } catch(Exception e) {
	        br.close();
	        subjectiveWriter.close();
	    }
	}
	
	// create gold standard
	public void readAutoReviewGoldSTD2(String fileName) throws IOException {
		PrintWriter subjectiveWriter = new PrintWriter("hierarchicaldata/merged.txt", "UTF-8");
	    BufferedReader br = new BufferedReader(new FileReader(fileName));
	    
	    try {
	    	
	        String line = br.readLine();
	        HashMap<Integer, String> annotation = new HashMap<>();
	        String mergedString = "";
	        String model = "";
	        
	        while (line != null) {
	        	
	        	// pre and post purchase
	        	int lastPositionOfReview = -1;
	        	boolean IsvalidData = false;
	        	lastPositionOfReview = line.lastIndexOf("pre_purchase");
	        	if(lastPositionOfReview > -1){
	        		IsvalidData = true;
	        		if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0){
	        			annotation.put(1, stem(line.substring(0, lastPositionOfReview-1).trim()));
	        			mergedString = mergedString + " " + stem(line.substring(0, lastPositionOfReview-1).trim());
	        		}
	        	}
	        	else if(line.lastIndexOf("post_purchase") > -1){
	        		lastPositionOfReview = line.lastIndexOf("post_purchase");
		        	if(lastPositionOfReview > -1){
		        		IsvalidData = true;
		        		if(stem(line.substring(0, lastPositionOfReview-1).trim()).length() > 0){
		        			annotation.put(2, stem(line.substring(0, lastPositionOfReview-1).trim()));
		        			mergedString = mergedString + " " + stem(line.substring(0, lastPositionOfReview-1).trim());
		        		}
		        	}
	        	}
	        	else if(line.length() > 2){
	        		model = line.trim();
	        	}
	        	
	        	
	        	// positive and negative
	        	if(IsvalidData){
	        		
	        	}
	        	else{
	        		if(annotation.size() == 1){
	        			subjectiveWriter.println("\n\n" + model);
	        			if(annotation.containsKey(1))
	        				subjectiveWriter.println("1 " + mergedString);
	        			else
	        				subjectiveWriter.println("2 " + mergedString);
	        		}
	        		else if(annotation.size() > 1){
	        			subjectiveWriter.println("\n\n" + model);
	        			subjectiveWriter.println("3 " + mergedString);
	        		}
	        		
	        		mergedString = "";
	        		annotation.clear();
	        	}
	        	
	            line = br.readLine();	            
	        }
	        
	        br.close();
	        subjectiveWriter.close();
	        
	    } catch(Exception e) {
	        br.close();
	        subjectiveWriter.close();
	    }
	}
		
	//  get random samples from a file
	public void writeRandomSample() throws Exception{		 			
		BufferedReader br = new BufferedReader(new FileReader("backup/kbb results in sentence level.txt"));
	    PrintWriter attWriter = new PrintWriter("backup/randkbb.txt", "UTF-8");
	    int counter = 0, target = 0;
	    Random rand = new Random();
	    try {
	        String line = br.readLine();
	        while (line != null) {
	        	if(counter % 5 == 0){
	        		target = counter + rand.nextInt(5);		        		
	        	}
	        	counter++;
	        	if(target == counter)
	        		attWriter.println(line);
	        	line = br.readLine();
	        }
	        
	        br.close();
	        attWriter.close();
	    } catch(Exception e) {
	        br.close();
	        attWriter.close();
	    }
	}
		
	// create gold standard
	public void saveReviewWithModel(String datasetName) throws Exception{		 			
		BufferedReader br1 = new BufferedReader(new FileReader("goldstandards/" + datasetName +"dataset.txt"));
		BufferedReader br2 = new BufferedReader(new FileReader("goldstandards/" + datasetName +"-review.txt"));
		BufferedReader br3 = new BufferedReader(new FileReader("goldstandards/" + datasetName +"-sentence.txt"));		
	    PrintWriter attWriter = new PrintWriter("goldstandards/" + datasetName +"Review.txt", "UTF-8");
	    PrintWriter attWriter1 = new PrintWriter("goldstandards/" + datasetName +"Sentence.txt", "UTF-8");
	    PrintWriter attWriter2 = new PrintWriter("goldstandards/" + datasetName +"Sentence1.txt", "UTF-8");
	    PrintWriter attWriter3 = new PrintWriter("goldstandards/" + datasetName +"Sentence3.txt", "UTF-8");
	    
	    InputStream is = new FileInputStream("binFiles/en-sent.bin");
		SentenceModel model = new SentenceModel(is);
		SentenceDetectorME sdetector = new SentenceDetectorME(model);	 
	    
	    int counter = 0;
	    HashMap<Integer, String> data = new HashMap<>();
	    HashMap<Integer, String> make = new HashMap<>();
	    
	    try {
	        String line1 = br1.readLine();
	        String line2 = br2.readLine();
	        while (line1 != null) {
	        	String [] goldstandards = line1.split(",");
	        	attWriter.println(("\n\n" + goldstandards[1] + " " + goldstandards[2]).replace("\"", ""));
	        	attWriter.println(line2);		        	
	        	data.put(counter, line2);
	        	make.put(counter, ("\n\n" + goldstandards[1] + " " + goldstandards[2]).replace("\"", ""));
	        	counter++;
	        	
	        	String sentences[] = sdetector.sentDetect(line2.substring(2));
	        	for(int j=0; j<sentences.length; j++){
	        		attWriter1.println(("\n\n" + goldstandards[1] + " " + goldstandards[2]).replace("\"", ""));		
	        		attWriter1.println(line2.substring(0,1) + " " + sentences[j]);	
	        		attWriter2.println(line2.substring(0,1) + " " + sentences[j]);
	        	}
	        	
	        	line1 = br1.readLine();
	        	line2 = br2.readLine();
	        }
	        
	        br1.close();
	        br2.close();
	        attWriter.close();
	        attWriter1.close();
	        attWriter2.close();
	        is.close();
	    } catch(Exception e) {
	        br1.close();
	        br2.close();
	        attWriter.close();
	        attWriter1.close();
	        attWriter2.close();
	        is.close();
	    }	
	    
	    try {
	        String line3 = br3.readLine();
	        while (line3 != null) {
	        	
	        	for(int i=0; i<data.size();i++){
	        		if(data.get(i).contains(line3.substring(2))){
	        			attWriter3.println(make.get(i));
	        			attWriter3.println(line3);
	        			break;
	        		}
	        	}
	        	
	        	line3 = br3.readLine();
	        }
	        
	        br3.close();
	        attWriter3.close();
	    } catch(Exception e) {
	        br3.close();
	        attWriter3.close();
	    }	    
	    
	}
	
	
	// create Weka compatible file
	public void makeWekaFileFormat(Instances trainData, String fileName,
			Vector<String> documents, Vector<Integer> docIDs, String dataFile, boolean withLIWC) throws Exception {
		
		Vector<String> attributeList = new Vector<String>();

		// create entity relation file format and saved it to LIWC folder
		PrintWriter writer = new PrintWriter("liwc/" + fileName + ".arff", "UTF-8");
		
		// write relation name
		writer.println("@RELATION	reviews\n");
		
		// write bag of words attributes		
		for (int i = 0; i < trainData.numAttributes()-1; i++) {
			String attribute = "attr_" + (i+1);		
			writer.println("@ATTRIBUTE" + "\t" + attribute + "\t" + "NUMERIC");				
			attributeList.add(trainData.attribute(i).name());
		}
		
		// write extra two features: question mark and exclamatory sign
	    writer.println("@ATTRIBUTE" + "\textra_question\t" + "NUMERIC");	
	    writer.println("@ATTRIBUTE" + "\textra_excla\t" + "NUMERIC");	
	    	
	   // container of LIWC
		ArrayList<LIWCPair> allLiwc = new ArrayList<LIWCPair>();
		Map<String, Integer> liwcCollection = new HashMap<String, Integer>();
	 			
	    if(withLIWC){
			
			// write liwc attributes after reading LIWC files
			BufferedReader br = new BufferedReader(new FileReader("liwc/LIWC2001WordStatOriginal.txt")); 
		    
		    try {
		    	
		        String feature = "";
		        String line = br.readLine();
	
		        while (line != null) {
		        	if(line.trim().contains("I.") || line.trim().contains("II.") || line.trim().contains("III.") || line.trim().contains("IV.")
		        			  || line.trim().contains("V.") || line.trim().contains("VI.") || line.trim().contains("VII.")  || line.trim().contains("VIII."))
		        	{
		        		// do nothing
		        	}
		        	else if(!line.trim().contains("(1)"))
		        	{
		        		if(!line.toLowerCase().trim().equals(""))
		        		{
			        		feature =  "liwc_" + line.toLowerCase().trim();
			        		feature = feature.replace(" ", "");
			        		writer.println("@ATTRIBUTE" + "\t" + feature + "\t" + "NUMERIC");
			        		liwcCollection.put(feature, 0);
		        		}
		        	}
		        	else if(line.trim().contains("(1)"))
		        	{
		        		
			            // create data structure with liwc wordlist
		        		LIWCPair newItem = new LIWCPair(feature, line.toLowerCase().trim().replace("(1)", ""));
		        		allLiwc.add(newItem);
		        	}
		        	
		        	line = br.readLine();
		        }
		        
		        br.close();
		        
		    } catch(Exception e) {
		        br.close();
		        writer.close();
		    }
	    }
			
	    
		// write class attribute
		writer.println();
		String strClass = "1";
		for(int idx=2; idx <= trainData.numClasses(); idx++)
			strClass = strClass + "," + idx;
		
		writer.print("@ATTRIBUTE	class	{" + strClass + "}");
		
		// write data for all attributes
		writer.println("\n\n@DATA");
		
		// now its time to set weight
		for (int j = 0; j < trainData.numInstances(); ++j){
			
			Instance inst = trainData.instance(j);
			
			for(int m = 0; m < trainData.numAttributes()-1; m++){				
				writer.print(inst.value(m) + ",");
			}
			
			// write question mark attribute
			if(documents.get(j).contains("?")){
				writer.print("1,");
			}
			else{
				writer.print("0,");
			}
			
			// write exclamatory attribute
			if(documents.get(j).contains("!")){
				writer.print("1,");
			}
			else{
				writer.print("0,");
			}
				
			// add weight for liwc collection
			@SuppressWarnings({ "unchecked", "rawtypes" })
			Map<String, Integer> newLiwc = new HashMap(liwcCollection);
			setLIWCWeight(inst, newLiwc, allLiwc, trainData);
			if(withLIWC){
				for (String key:newLiwc.keySet()) {		
					writer.print(newLiwc.get(key) + ",");
				}
			}
			
			// write the label for this instance
			writer.println(inst.classAttribute().value((int)(inst.classValue())));
			System.out.println("Processed "+ j + " out of " + trainData.numInstances() + " instances. ");
		}
		
		writer.close();	
		
	}
	
	// set LIWC feature value
    public void setLIWCWeight(Instance inst, Map<String, Integer> newLiwc, ArrayList<LIWCPair> allLiwc, Instances trainData){
		
		// get attribute index list
		ArrayList<String> attributeIndex = new ArrayList<String>();
		String instanceString = inst.toString();
		instanceString = instanceString.replace("{", "");
		instanceString = instanceString.replace("}", "");
		String[] arrAttributeIndex = instanceString.split(","); 
		//System.out.println(instanceString);
		
		for(int idx = 0; idx < inst.numValues(); idx++){
			int index = Integer.parseInt(arrAttributeIndex[idx].split(" ")[0].trim());
			Attribute attr = inst.attribute(index);
			String attrName = attr.name().trim().toLowerCase();
			attributeIndex.add(idx, attrName);
		}
		
		for(int i=0; i < allLiwc.size(); i++){
			LIWCPair pair = allLiwc.get(i);
			
			String liwcFeature = pair.featureValue.toLowerCase();
			liwcFeature = liwcFeature.replace("*", "");
			liwcFeature = liwcFeature.replace("(1)", "").trim();
			
			for(int j = 0; j < attributeIndex.size(); j++){
				//System.out.println(liwcFeature);
				if(liwcFeature.equals(attributeIndex.get(j).trim())){
					// sum up weight for each attribute 
					newLiwc.put(pair.feature, newLiwc.get(pair.feature)+1);
				}
			}
		}
	}
}
