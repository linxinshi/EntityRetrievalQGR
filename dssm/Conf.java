public class Conf {
	
	public String sent2vec_home = "C:/Users/kplai/Downloads/Sent2Vec";
	
	
	//sent2vec train
	public String sent2vec_train_exe = String.format("%s/sample/bin/DSSM_Train.exe", sent2vec_home);
	public String sent2vec_train_folder = String.format("%s/tmpfolder/train", sent2vec_home);
	
	public String sent2vec_train_conf_path = String.format("%s/train_conf.txt", sent2vec_train_folder);
	public String sent2vec_train_model_prefix = String.format("%s/sent2vec_model", sent2vec_train_folder);
	public String sent2vec_train_log = String.format("%s/log.txt", sent2vec_train_folder);

	public String sent2vec_train_wordhash_exe = String.format("%s/sample/bin/WordHash.exe", sent2vec_home);
	public String sent2vec_train_wordhash_fea_prefix = String.format("%s/train_", sent2vec_train_folder);
	public String sent2vec_train_wordhash_src_bin = String.format("%s/train_src_seq_fea.bin", sent2vec_train_folder);
	public String sent2vec_train_wordhash_tgt_bin = String.format("%s/train_tgt_seq_fea.bin", sent2vec_train_folder);
	
	public String sent2vec_train_nce_exe = String.format("%s/sample/bin/ComputelogPD.exe", sent2vec_home);
	public String sent2vec_train_nce_logpd_prefix = String.format("%s/train_logpD_s", sent2vec_train_folder);
	
	//sent2vec predict
	public String sent2vec_predict_exe = String.format("%s/sample/bin/sent2vec.exe", sent2vec_home);
	public String sent2vec_predict_folder = String.format("%s/tmpfolder/predict", sent2vec_home);
	
//	public String sent2vec_predict_infile = String.format("%s/_sent2vec_infile.txt", sent2vec_predict_folder);
	public String sent2vec_predict_outfile_prefix = String.format("%s/_sent2vec_output", sent2vec_predict_folder);
	
	public String sent2vec_src_model = String.format("%s_QUERY_DONE", sent2vec_train_model_prefix);
//	public String sent2vec_src_vocab = String.format("%s/sent2vec_src_vocab", sent2vec_train_folder);
	public String sent2vec_tgt_model = String.format("%s_DOC_DONE", sent2vec_train_model_prefix);
//	public String sent2vec_tgt_vocab = String.format("%s/sent2vec_tgt_vocab", sent2vec_train_folder);
	
	public Conf(String home) {
		if (home == null)
			return;
		
		this.sent2vec_home = home;
		
		//sent2vec train
		sent2vec_train_exe = String.format("%s/sample/bin/DSSM_Train.exe", sent2vec_home);
		sent2vec_train_folder = String.format("%s/tmpfolder/train", sent2vec_home);
		
		sent2vec_train_conf_path = String.format("%s/train_conf.txt", sent2vec_train_folder);
		sent2vec_train_model_prefix = String.format("%s/sent2vec_model", sent2vec_train_folder);
		sent2vec_train_log = String.format("%s/log.txt", sent2vec_train_folder);
		
		sent2vec_train_wordhash_exe = String.format("%s/sample/bin/WordHash.exe", sent2vec_home);
		sent2vec_train_wordhash_fea_prefix = String.format("%s/train_", sent2vec_train_folder);
		sent2vec_train_wordhash_src_bin = String.format("%s/train_src_seq_fea.bin", sent2vec_train_folder);
		sent2vec_train_wordhash_tgt_bin = String.format("%s/train_tgt_seq_fea.bin", sent2vec_train_folder);
		
		sent2vec_train_nce_exe = String.format("%s/sample/bin/ComputelogPD.exe", sent2vec_home);
		sent2vec_train_nce_logpd_prefix = String.format("%s/train_logpD_s", sent2vec_train_folder);
		
		//sent2vec predict
		sent2vec_predict_exe = String.format("%s/sample/bin/sent2vec.exe", sent2vec_home);
		sent2vec_predict_folder = String.format("%s/tmpfolder/predict", sent2vec_home);
		
//		sent2vec_predict_infile = String.format("%s/_sent2vec_infile.txt", sent2vec_predict_folder);
		sent2vec_predict_outfile_prefix = String.format("%s/_sent2vec_output", sent2vec_predict_folder);
		
		sent2vec_src_model = String.format("%s_QUERY_DONE", sent2vec_train_model_prefix);
//		sent2vec_src_vocab = String.format("%s/sent2vec_src_vocab", sent2vec_train_folder);
		sent2vec_tgt_model = String.format("%s_DOC_DONE", sent2vec_train_model_prefix);
//		sent2vec_tgt_vocab = String.format("%s/sent2vec_tgt_vocab", sent2vec_train_folder);
	}
}
