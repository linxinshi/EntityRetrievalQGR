import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Sent2VecHelper {
	
	private Conf conf;
	
	private String sent2vec_train_exe;
	private String sent2vec_train_conf_path;
	private String sent2vec_train_model_prefix;
	private String sent2vec_train_log;

	private String sent2vec_train_wordhash_exe;
	private String sent2vec_train_wordhash_fea_prefix;
	private String sent2vec_train_wordhash_src_bin;
	private String sent2vec_train_wordhash_tgt_bin;
	
	
	private String sent2vec_train_nce_exe;
	private String sent2vec_train_nce_logpd_prefix;
	
	private String sent2vec_predict_exe;
	private String sent2vec_src_model;
	private String sent2vec_tgt_model;
	private String sent2vec_predict_infile;
	private String sent2vec_predict_outfile_prefix;
	
	private File train_folder;
	private File predict_folder;
	
	public enum Sent2Vec_Model_Type {
		DSSM,
		CDSSM,
		MIX
	}
	
	public Sent2VecHelper(Conf conf) {
		if (conf == null)
			conf = new Conf(null);
		
		this.conf = conf;
		
		//train
		this.sent2vec_train_exe = conf.sent2vec_train_exe;
		this.train_folder = new File(conf.sent2vec_train_folder);
		if (!this.train_folder.exists())
			this.train_folder.mkdirs();
		this.sent2vec_train_model_prefix = conf.sent2vec_train_model_prefix;
		this.sent2vec_train_log = conf.sent2vec_train_log;
		
		this.sent2vec_train_conf_path = conf.sent2vec_train_conf_path;
		
		this.sent2vec_train_wordhash_exe = conf.sent2vec_train_wordhash_exe;
		this.sent2vec_train_wordhash_fea_prefix = conf.sent2vec_train_wordhash_fea_prefix;
		this.sent2vec_train_wordhash_src_bin = conf.sent2vec_train_wordhash_src_bin;
		this.sent2vec_train_wordhash_tgt_bin = conf.sent2vec_train_wordhash_tgt_bin;
		
		
		this.sent2vec_train_nce_exe = conf.sent2vec_train_nce_exe;
		this.sent2vec_train_nce_logpd_prefix = conf.sent2vec_train_nce_logpd_prefix;
		
		//predict
		this.sent2vec_predict_exe = conf.sent2vec_predict_exe;
		this.predict_folder = new File(conf.sent2vec_predict_folder);
		if (!this.predict_folder.exists())
			this.predict_folder.mkdirs();
		
		this.sent2vec_predict_outfile_prefix = conf.sent2vec_predict_outfile_prefix;
		
		
		//todo: set the path for both src and tgt model and vocab after doing training
		this.sent2vec_src_model = conf.sent2vec_src_model;
		this.sent2vec_tgt_model = conf.sent2vec_tgt_model;
	}
	
	private void sent2vec_predict(Sent2Vec_Model_Type type,
	                          String inSrcModel,
	                          String inSrcVocab,
	                          String inTgtModel,
	                          String inTgtVocab,
	                          String inFilename,
	                          String outFilenamePrefix) {
		try {
			
			if (type != Sent2Vec_Model_Type.MIX)
				return;
			
			if (inSrcModel == null)
				inSrcModel = sent2vec_src_model;
			if (inSrcVocab == null) {
				System.out.println("sent2vec_predict: please provide a valid source vocabulary file (tri-gram file)");
				return;
			}
			if (inTgtModel == null)
				inTgtModel = sent2vec_tgt_model;
			if (inTgtVocab == null) {
				System.out.println("sent2vec_predict: please provide a valid target vocabulary file (tri-gram file)");
				return;
			}
			
			if (inFilename == null) {
				System.out.println("sent2vec_predict: please provide a valid input file (training pair token file)");
				return;
			}
			if (outFilenamePrefix == null)
				outFilenamePrefix = sent2vec_predict_outfile_prefix;
			
			StringBuilder cmd = new StringBuilder();
			cmd.append(sent2vec_predict_exe);
			
			cmd.append(String.format(" /inSrcModel %s", inSrcModel)); //mixmodelmodel_QUERY_DONE
			cmd.append(String.format(" /inSrcVocab %s", inSrcVocab)); //datal3g.txt
			cmd.append(String.format(" /inSrcModelType CDSSM"));
			cmd.append(String.format(" /inSrcMaxRetainedSeqLength 20"));
			cmd.append(String.format(" /inTgtModel %s", inTgtModel)); //mixmodelmodel_DOC_DONE
			cmd.append(String.format(" /inTgtVocab %s", inTgtVocab)); //datal3g.txt
			cmd.append(String.format(" /inTgtModelType DSSM"));
			cmd.append(String.format(" /inFilename %s", inFilename)); //datadev.pair.tok.tsv
			cmd.append(String.format(" /outFilenamePrefix %s", outFilenamePrefix)); //mixdev.sent2vec
			
//			Process process = new ProcessBuilder(sent2vec_predict_exe,
//
//					String.format("/inSrcModel %s", inSrcModel), //mixmodelmodel_QUERY_DONE
//					String.format("/inSrcVocab %s", inSrcVocab), //datal3g.txt
//					String.format("/inSrcModelType CDSSM"),
//					String.format("/inSrcMaxRetainedSeqLength 20"),
//					String.format("/inTgtModel %s", inTgtModel), //mixmodelmodel_DOC_DONE
//					String.format("/inTgtVocab %s", inTgtVocab), //datal3g.txt
//					String.format("/inTgtModelType DSSM"),
//					String.format("/inFilename %s", inFilename), //datadev.pair.tok.tsv
//					String.format("/outFilenamePrefix %s", outFilenamePrefix) //mixdev.sent2vec
//
//					).start();
			
			String[] cmds = cmd.toString().split(" ");
			
			Process process = new ProcessBuilder(cmds).start();
			
			process.waitFor();
			
			BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String tmps = null;
			while ((tmps = brdr.readLine()) != null) {
				System.out.println(tmps);
			}
			brdr.close();
			
			process.destroy();
			
			System.out.println(String.format("Output location: %s", predict_folder));
					
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void prepare_sent2vec_input_file(List<String> test_q, List<String> preds) {
		List<String> rs = new ArrayList<>(test_q.size() * preds.size());
		
		for (String q : test_q) {
			for (String pred : preds) {
				rs.add(String.format("%s\t%s", q, pred));
			}
		}
		
		MyFIO.write_line_file(sent2vec_predict_infile, rs);
	}
	
	public void predict() {
	
	}
	
	public String prepare_feature_file(String infile, String src_vocab, String tgt_vocab, int max_word, String outfile_prefix) {
		//..\bin\WordHash.exe --pair2seqfea data\train.pair.tok.tsv data\l3g.txt data\l3g.txt 1 mix\train.1
		
		try {
			
			if (infile == null) {
				System.out.println("prepare_feature_file: please provide a valid input file (training pair token file)");
				return null;
			}
			
			if (src_vocab == null) {
				System.out.println("prepare_feature_file: please provide a valid source vocabulary file (tri-gram file)");
				return null;
			}
			if (tgt_vocab == null) {
				System.out.println("prepare_feature_file: please provide a valid target vocabulary file (tri-gram file)");
				return null;
			}
			
			if (max_word < 1) {
				System.out.println("prepare_feature_file: max word should be greater than 0");
				return null;
			}
			
			if (outfile_prefix == null)
				outfile_prefix = sent2vec_train_wordhash_fea_prefix;
			
			outfile_prefix = String.format("%s%d", outfile_prefix, max_word);
			
			String cmd = String.format("%s --pair2seqfea %s %s %s %d %s",
					sent2vec_train_wordhash_exe, infile, src_vocab, tgt_vocab, max_word, outfile_prefix);
			String[] cmds = cmd.split(" ");
			
			Process process = new ProcessBuilder(cmds).start();
			
			process.waitFor();
			
			BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String tmps = null;
			while ((tmps = brdr.readLine()) != null) {
				System.out.println(tmps);
			}
			brdr.close();
			
			process.destroy();
			
			return outfile_prefix;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public String prepare_binary_file(String infile, int batch_size, String outfile, boolean srcfile) {
		//..\bin\WordHash.exe --seqfea2bin  mix\train.20.src.seq.fea 1024 mix\train.src.seq.fea.bin
		
		try {
			
			if (infile == null) {
				System.out.println("prepare_binary_file: please provide a valid input file (sequence feature file)");
				return null;
			}
			
			if (outfile == null) {
				if (srcfile)
					outfile = sent2vec_train_wordhash_src_bin;
				else
					outfile = sent2vec_train_wordhash_tgt_bin;
			}
			
			if (batch_size < 1)
				batch_size = 1024;
			
			String cmd = String.format("%s --seqfea2bin %s %d %s",
					sent2vec_train_wordhash_exe, infile, batch_size, outfile);
			
			String[] cmds = cmd.split(" ");
			
			Process process = new ProcessBuilder(cmds).start();
			
			process.waitFor();
			
			BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String tmps = null;
			while ((tmps = brdr.readLine()) != null) {
				System.out.println(tmps);
			}
			brdr.close();
			
			process.destroy();
			return outfile;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public String prepare_nce_file(String infile, String outfile_prefix, double scale) {
		//..\bin\ComputelogPD.exe /i data\train.pair.tok.tsv /o mix\train.logpD.s75 /C 1 /S 0.75
		
		try {
			
			if (infile == null) {
				System.out.println("prepare_nce_file: please provide a valid input file (training pair token file)");
				return null;
			}
			
			if (outfile_prefix == null)
				outfile_prefix = sent2vec_train_nce_logpd_prefix;
			
			if (scale <= 0 || scale > 1)
				scale = 0.75;
			
			String outfile = String.format("%s%.2f",outfile_prefix, scale);
			
			String cmd = String.format("%s /i %s /o %s /C 1 /S %f",
					sent2vec_train_nce_exe, infile, outfile, scale);
					
			String[] cmds = cmd.split(" ");
			
			Process process = new ProcessBuilder(cmds).start();
			MyUtil.LogStreamReader lsr = new MyUtil.LogStreamReader(process.getInputStream());
			Thread thread = new Thread(lsr, "LogStreamReader");
			thread.start();
			
			process.waitFor();
			
//			BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
//			String tmps = null;
//			while ((tmps = brdr.readLine()) != null) {
//				System.out.println(tmps);
//			}
//			brdr.close();
			
			process.destroy();
			
			return outfile;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public String prepare_train_conf_file(String oconf_path,
	                                    String sent2vec_train_model_prefix,
	                                    String sent2vec_train_log,
	                                    
	                                    String sent2vec_train_src_bin,
	                                    String sent2vec_train_tgt_bin,
	                                    
	                                    String sent2vec_train_nce_logpd,
	                                    
	                                    int batch_size,
	                                    int iter,
	                                    double learning_rate) {
		
		BufferedWriter bwtr = null;
		try {
			
			if (oconf_path == null)
				oconf_path = sent2vec_train_conf_path;
			
			bwtr = MyFIO.myWriter(oconf_path, null, false);
			
			if (sent2vec_train_model_prefix == null)
				sent2vec_train_model_prefix = this.sent2vec_train_model_prefix;
			if (sent2vec_train_log == null)
				sent2vec_train_log = this.sent2vec_train_log;
			
			if (sent2vec_train_src_bin == null)
				sent2vec_train_src_bin = sent2vec_train_wordhash_src_bin;
			if (sent2vec_train_tgt_bin == null)
				sent2vec_train_tgt_bin = sent2vec_train_wordhash_tgt_bin;
			
			if (sent2vec_train_nce_logpd == null) {
				System.out.println("prepare_train_conf_file: please provide a valid nce logpd file");
				return null;
			}
			
			
			if (batch_size < 1)
				batch_size = 1024;
			if (iter <= 0)
				iter = 500;
			if (learning_rate <= 0)
				learning_rate = 0.02;
			
			bwtr.write(String.format("CUBLAS\t1\n"));
			bwtr.write(String.format("OBJECTIVE\tNCE\n"));
			bwtr.write(String.format("LOSS_REPORT\t1\n"));
			bwtr.write(String.format("MODELPATH\t%s\n", sent2vec_train_model_prefix));
			bwtr.write(String.format("LOGFILE\t%s\n", sent2vec_train_log));
			bwtr.write(String.format("QFILE\t%s\n", sent2vec_train_src_bin));
			bwtr.write(String.format("DFILE\t%s\n", sent2vec_train_tgt_bin));
			bwtr.write(String.format("NCE_PROB_FILE\t%s\n", sent2vec_train_nce_logpd));
			bwtr.write(String.format("BATCHSIZE\t%d\n", batch_size));
			bwtr.write(String.format("NTRIAL\t50\n"));
			bwtr.write(String.format("MAX_ITER\t%d\n", iter));
			bwtr.write(String.format("PARM_GAMMA\t25\n"));
			bwtr.write(String.format("TRAIN_TEST_RATE\t1\n"));
			bwtr.write(String.format("LEARNINGRATE\t%f\n", learning_rate));
			bwtr.write(String.format("SOURCE_LAYER_DIM\t1000,300\n"));
			bwtr.write(String.format("SOURCE_LAYERWEIGHT_SIGMA\t0.1,0.1\n"));
			bwtr.write(String.format("SOURCE_ACTIVATION\t1,1\t#0: Linear   1: Tanh    2: rectified\n"));
			bwtr.write(String.format("SOURCE_ARCH\t1,0\t#0: Fully Connected\t1: Convolutional\n"));
			bwtr.write(String.format("SOURCE_ARCH_WIND\t3,1\n"));
			bwtr.write(String.format("TARGET_LAYER_DIM\t1000,300\n"));
			bwtr.write(String.format("TARGET_LAYERWEIGHT_SIGMA\t0.1,0.1\n"));
			bwtr.write(String.format("TARGET_ACTIVATION\t1,1\n"));
			bwtr.write(String.format("TARGET_ARCH\t0,0\n"));
			bwtr.write(String.format("TARGET_ARCH_WIND\t1,1\n"));
			bwtr.write(String.format("MIRROR_INIT\t0\n"));
			bwtr.write(String.format("DEVICE\t1\n"));
			bwtr.write(String.format("REJECT_RATE\t1.0\n"));
			bwtr.write(String.format("DOWN_RATE\t1.0\n"));
			bwtr.write(String.format("ACCEPT_RANGE\t1.0\n"));
			bwtr.write(String.format("MATH_LIB\tGPU\n"));
			
			bwtr.close();
			
			return oconf_path;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public void sent2vec_train(String conf_path) {
//		..\bin\DSSM_Train.exe mix.config.txt
		
		try {
			if (conf_path == null)
				conf_path = sent2vec_train_conf_path;
			
			System.out.println(String.format("%s %s", sent2vec_train_exe, conf_path));
			
			Process process = new ProcessBuilder(sent2vec_train_exe, conf_path).start();
			MyUtil.LogStreamReader lsr = new MyUtil.LogStreamReader(process.getInputStream());
			Thread thread = new Thread(lsr, "LogStreamReader");
			thread.start();
			
			process.waitFor();
			
//			BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
//			String tmps = null;
//			while ((tmps = brdr.readLine()) != null) {
//				System.out.println(tmps);
//			}
//			brdr.close();
//
			process.destroy();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void train(String training_data_file, String src_vocab_file, String tgt_vocab_file) {
		int batch_size = 1024;
		
		String feature1 = prepare_feature_file(training_data_file, src_vocab_file, tgt_vocab_file, 1, null);
		String feature20 = prepare_feature_file(training_data_file, src_vocab_file, tgt_vocab_file, 20, null);
		
		String src_feature = String.format("%s.src.seq.fea", feature20);
		String tgt_feature = String.format("%s.tgt.seq.fea", feature1);
		
		String src_bin = prepare_binary_file(src_feature, batch_size, null, true);
		String tgt_bin = prepare_binary_file(tgt_feature, batch_size, null, false);
		
		String nce_logpd = prepare_nce_file(training_data_file, null, 0.75);
		prepare_train_conf_file(null, null, null, src_bin, tgt_bin, nce_logpd, batch_size, 500, 0.02);
		sent2vec_train(null);
	}
	
	public static void main(String[] args) {
		
		String sent2vec_location = "C:\\Users\\kplai\\Downloads\\Sent2Vec";
		String training_data_file = "C:\\Users\\kplai\\Downloads\\Sent2Vec\\sample\\training\\data\\train.pair.tok.tsv";
		String src_vocab = "C:\\Users\\kplai\\Downloads\\Sent2Vec\\sample\\training\\data\\l3g.txt";
		String tgt_vocab = "C:\\Users\\kplai\\Downloads\\Sent2Vec\\sample\\training\\data\\l3g.txt";
		
		String testing_data_file = "C:\\Users\\kplai\\Downloads\\Sent2Vec\\sample\\training\\data\\dev.pair.tok.tsv";
		
		Conf conf = new Conf(sent2vec_location);
		Sent2VecHelper helper = new Sent2VecHelper(conf);
		
		helper.train(training_data_file, src_vocab, tgt_vocab);
		helper.sent2vec_predict(Sent2Vec_Model_Type.MIX, null, src_vocab, null, tgt_vocab, testing_data_file, null);
	}
}