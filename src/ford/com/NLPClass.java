package ford.com;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


import opennlp.tools.cmdline.postag.POSModelLoader;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

public class NLPClass {
	
	// Performed part of speech tagging
	public void doPOSTagging(String dataFile) throws IOException {
		
		int counter = 1;
		
		POSModel model = new POSModelLoader().load(new File("binFiles/en-pos-maxent.bin"));
		POSTaggerME tagger = new POSTaggerME(model);
		
		BufferedReader br = new BufferedReader(new FileReader("data/" + dataFile + ".txt"));
	    PrintWriter attWriterParse = new PrintWriter("nlpdata/" + dataFile + "_parse.txt", "UTF-8");
	    PrintWriter attWriterTag = new PrintWriter("nlpdata/" + dataFile + "_tag.txt", "UTF-8");
	    
	    try {

	    	String lineData = br.readLine();
	        while (lineData != null) {
		        
		        String input = lineData.substring(2);
				ObjectStream<String> lineStream = new PlainTextByLineStream(
						new StringReader(input));

				StringBuilder inst = new StringBuilder();

				String line;
				
				while ((line = lineStream.read()) != null) {

					String whitespaceTokenizerLine[] = WhitespaceTokenizer.INSTANCE
							.tokenize(line);
					String[] tags = tagger.tag(whitespaceTokenizerLine);

					POSSample sample = new POSSample(whitespaceTokenizerLine, tags);
					inst.append(sample.toString() + " ");
				}
				
				attWriterTag.println(inst.toString()); 				
				lineData = br.readLine();				
				System.out.println("Processed " + counter);
				counter++;
	        }
	        
	        br.close();
	        attWriterTag.close();
	        attWriterParse.close();
	    } catch(Exception e) {
	        br.close();
	        attWriterTag.close();
	        attWriterParse.close();
	    }
	}
	
	// for debugging purpose
	public void displayPOSTagOnly(String dataFile) throws IOException {
	    BufferedReader purchaseReader = new BufferedReader(new FileReader("data/" + dataFile + ".txt"));
	    BufferedReader tagReader = new BufferedReader(new FileReader("nlpdata/" + dataFile + "_tag.txt"));
	    PrintWriter attWriter = new PrintWriter("nlpdata/" + dataFile + "_tagonly.txt", "UTF-8");
	    
	    try {
	        String line = tagReader.readLine();
	        String sample = purchaseReader.readLine();
	        
	        while (line != null) {
	        	String []words = line.split(" ");
	        	StringBuilder sb = new StringBuilder();
	        	sb.append(sample.split(" ")[0].trim()+ " ");
	        	
	        	for(int i = 0; i < words.length; i++){
	        		String []tagWords = words[i].split("_");
	        		sb.append(tagWords[1].trim()+" ");
	        		
	        	}
	        	attWriter.println(sb.toString());   
	        	line = tagReader.readLine();
	        	sample = purchaseReader.readLine();
	        }
	        
	        tagReader.close();
	        purchaseReader.close();
	        attWriter.close();
	    } catch(Exception e) {
	    	tagReader.close();
	    	purchaseReader.close();
	        attWriter.close();
	    }
	}
	
	// for debugging purpose
	public void generatePatterns(String dataFile) throws IOException {
	    BufferedReader tagReader1 = new BufferedReader(new FileReader("nlpdata/" + dataFile + "_tagonly_split1.txt"));
	    BufferedReader tagReader2 = new BufferedReader(new FileReader("nlpdata/" + dataFile + "_tagonly_split2.txt"));
	    PrintWriter patWriter1 = new PrintWriter("nlpdata/split1_pattern.txt", "UTF-8");
	    PrintWriter patWriter2 = new PrintWriter("nlpdata/split2_pattern.txt", "UTF-8");
	    HashMap <String, Integer> voabPattern1 = new HashMap<String, Integer>();
	    HashMap <String, Integer> voabPattern2 = new HashMap<String, Integer>();
	    
	    try {
	        String line = tagReader1.readLine();
	        
	        while (line != null) {
	        	String []words = line.split(" ");
	        	
	        	if(words.length > 2){
		        	for(int i = 1; i < words.length-1; i++){
		        		StringBuilder patternText = new StringBuilder();
		        		patternText.append(words[i]);
		        		patternText.append(" ");
		        		patternText.append(words[i+1]);
		        		//patternText.append(" ");
		        		//patternText.append(words[i+2]);
		        		if(voabPattern1.containsKey(patternText.toString().trim())){
		        			voabPattern1.put(patternText.toString().trim(), voabPattern1.get(patternText.toString().trim())+1);
		        		}
		        		else{
		        			voabPattern1.put(patternText.toString().trim(), 1);
		        		}
		        		
		        	}
	        	} 
	        	line = tagReader1.readLine();
	        }
	        
	        tagReader1.close();
	    } catch(Exception e) {
	    	tagReader1.close();
	    }
	    
	    try {
	        String line = tagReader2.readLine();
	        
	        while (line != null) {
	        	String []words = line.split(" ");
	        	
	        	if(words.length > 2){
		        	for(int i = 1; i < words.length-1; i++){
		        		StringBuilder patternText = new StringBuilder();
		        		patternText.append(words[i]);
		        		patternText.append(" ");
		        		patternText.append(words[i+1]);
		        		//patternText.append(" ");
		        		//patternText.append(words[i+2]);
		        		if(voabPattern2.containsKey(patternText.toString().trim())){
		        			voabPattern2.put(patternText.toString().trim(), voabPattern2.get(patternText.toString().trim())+1);
		        		}
		        		else{
		        			voabPattern2.put(patternText.toString().trim(), 1);
		        		}
		        		
		        	}
	        	} 
	        	line = tagReader2.readLine();
	        }
	        
	        tagReader2.close();
	    } catch(Exception e) {
	    	tagReader2.close();
	    }
	    
	    HashMap <String, Integer> voabFinalPattern1 = new HashMap<String, Integer>();
	    HashMap <String, Integer> voabFinalPattern2 = new HashMap<String, Integer>();
	    
	    for (String key:voabPattern1.keySet()) {
	    	if(!voabPattern2.containsKey(key)){
	    		voabFinalPattern1.put(key, voabPattern1.get(key));
	    	}
	    	else if(voabPattern1.get(key)/voabPattern2.get(key) > 2){
	    		voabFinalPattern1.put(key, voabPattern1.get(key));
	    	}
	    }
	    
	    for (String key:voabPattern2.keySet()) {
	    	if(!voabPattern1.containsKey(key)){
	    		voabFinalPattern2.put(key, voabPattern2.get(key));
	    	}
	    	else if(voabPattern2.get(key)/voabPattern1.get(key) > 2){
	    		voabFinalPattern2.put(key, voabPattern2.get(key));
	    	}
	    }
	    
	    HashMap <String, Integer> voabSortPattern1 = sortByValues(voabFinalPattern1);
	    HashMap <String, Integer> voabSortPattern2 = sortByValues(voabFinalPattern2);
	    
	    System.out.println(voabSortPattern1);
	    System.out.println(voabSortPattern2);
	}
	
	// for research purpose
	@SuppressWarnings("unchecked")
	private HashMap sortByValues(HashMap map) { 
	       List list = new LinkedList(map.entrySet());
	       // Defined Custom Comparator here
	       Collections.sort(list, new Comparator() {
	            public int compare(Object o1, Object o2) {
	               return ((Comparable) ((Map.Entry) (o2)).getValue())
	                  .compareTo(((Map.Entry) (o1)).getValue());
	            }
	       });

	       // Here I am copying the sorted list in HashMap
	       // using LinkedHashMap to preserve the insertion order
	       HashMap sortedHashMap = new LinkedHashMap();
	       for (Iterator it = list.iterator(); it.hasNext();) {
	              Map.Entry entry = (Map.Entry) it.next();
	              sortedHashMap.put(entry.getKey(), entry.getValue());
	       } 
	       return sortedHashMap;
	  }
}
