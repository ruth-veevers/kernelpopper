
project_dir = projectDir

process one_cubic {
	publishDir "${project_dir}", mode: 'copy', overwrite: true
	label 'usearray'
	label 'largemem'
	input:
	tuple val(n_features), val(seed)
	
	output:
	path "result_cubic_${n_features}_${seed}.txt"
	
	script:
	"""
	python ${project_dir}/scripts.py c $n_features $seed
	"""
}

workflow {
	main:
		feature_channel=Channel.of(0..149)
		seed_channel=Channel.of(0..9)
		one_cubic(feature_channel.combine(seed_channel))
}