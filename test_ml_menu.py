import unittest
from pathlib import Path
from unittest.mock import patch

import ml_cleanup
import ml_menu


class MenuTest(unittest.TestCase):
    @patch('ml_menu.run')
    @patch('builtins.input', side_effect=['2', 'n', '8', 'y', '0'])
    def test_ubuntu_install_uses_existing_installer(self, _input, run):
        with patch.object(ml_menu, 'ROOT', Path('/opt/machine_learning')), patch.object(ml_menu.sys, 'platform', 'linux'):
            ml_menu.main()
        self.assertEqual(run.call_args.args[0][:2], ['sudo', 'bash'])
        self.assertTrue(run.call_args.args[0][2].endswith('deploy\\install-ubuntu.sh') or
                        run.call_args.args[0][2].endswith('deploy/install-ubuntu.sh'))

    @patch('ml_menu.run')
    @patch('builtins.input', side_effect=['1', 'n', '7', '5', '0'])
    def test_cleanup_runs_in_selected_environment(self, _input, run):
        ml_menu.main()
        self.assertEqual(run.call_args.args[0][-2:], ['ml_cleanup.py', 'seeded-histories'])

    @patch('ml_menu.run')
    @patch('builtins.input', side_effect=['1', 'y', '4', '0'])
    def test_combined_docker_run_stops_on_refresh_failure(self, _input, run):
        run.return_value = False
        ml_menu.main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], [
            'docker', 'compose', 'run', '--rm', '-e', 'ML_INCLUDE_SEEDED_DATA=true',
            'recommender-trainer', 'python', 'snapshot_user_histories.py',
        ])

    @patch('ml_menu.run')
    @patch('builtins.input', side_effect=['2', 'n', '4', '0'])
    def test_combined_native_run_calls_scripts_in_order(self, _input, run):
        run.return_value = True
        ml_menu.main()
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], [
            'snapshot_user_histories.py', 'train_svd_recommender.py', 'evaluate_recommender.py',
        ])

    def test_file_cleanup_requires_confirmation(self):
        with patch('sys.argv', ['ml_cleanup.py', 'model']), patch.object(Path, 'unlink') as unlink:
            with patch('builtins.input', return_value='no'):
                ml_cleanup.main()
            unlink.assert_not_called()
            with patch('builtins.input', return_value='DELETE'):
                ml_cleanup.main()
            unlink.assert_called_once_with(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
