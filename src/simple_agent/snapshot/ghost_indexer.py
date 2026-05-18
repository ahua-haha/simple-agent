import os
from git import Repo, GitCommandError

class RepoWatcher:
    def __init__(self, project_root, metadata_dir):
        """
        :param project_root: The actual source code directory.
        :param shadow_meta_dir: Where the 'ghost' .git and indices will live.
        """
        self.project_root = os.path.abspath(project_root)
        self.metadata_dir = os.path.abspath(metadata_dir)
        self.git_dir = os.path.join(self.metadata_dir, "git")
        self.index_file = os.path.join(self.metadata_dir, "index")
        
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize a bare repo to serve as our private object database
        if not os.path.exists(self.git_dir):
            self.repo = Repo.init(self.git_dir, bare=True)
        else:
            self.repo = Repo(self.git_dir)

        # Ensure the shadow repo doesn't track the actual .git folder
        self._exclude_real_git()

    def _exclude_real_git(self):
        exclude_path = os.path.join(self.git_dir, "info", "exclude")
        os.makedirs(os.path.dirname(exclude_path), exist_ok=True)
        with open(exclude_path, "w") as f:
            f.write(".git/\n")

    def _get_env(self):
        """Standard environment for isolated plumbing operations."""
        return {
            "GIT_DIR": self.git_dir,
            "GIT_WORK_TREE": self.project_root,
            "GIT_INDEX_FILE": self.index_file
        }

    def take_snapshot(self):
        """
        Stages all non-ignored files and records the directory structure.
        :return: SHA-1 Tree Hash
        """
        env = self._get_env()
        try:
            with self.repo.git.custom_environment(**env):
                # Git automatically respects .gitignore in project_root
                self.repo.git.add(A=True)
                # 'write-tree' creates the snapshot hash
                return self.repo.git.write_tree()
        except GitCommandError as e:
            print(f"Snapshot error: {e}")
            return None

    def get_diff(self, old_hash, new_hash):
        """Compares two Tree Hashes and returns the patch."""
        env = self._get_env()
        with self.repo.git.custom_environment(**env):
            return self.repo.git.diff(old_hash, new_hash)

    def get_file_diff(self, old_hash, new_hash, path, context_lines=3):
        """Compares two Tree Hashes for a single file and returns the patch."""
        env = self._get_env()
        absolute_path = os.path.join(self.project_root, path)
        with self.repo.git.custom_environment(**env):
            return self.repo.git.diff(old_hash, new_hash, f"-U{context_lines}", "--", absolute_path)

    def get_changed_files(self, old_hash, new_hash):
        """List changed files with their lifecycle state (A, M, D)."""
        env = self._get_env()
        with self.repo.git.custom_environment(**env):
            return self.repo.git.diff("--name-status", old_hash, new_hash)

    def get_changed_files_with_rename(self, old_hash, new_hash):
        """List changed files with rename detection.

        Returns a list of ``(status, old_path, new_path_or_None)`` tuples.
        Renames (status ``R``) include the new path; other statuses have
        ``None`` in the third position.
        """
        env = self._get_env()
        with self.repo.git.custom_environment(**env):
            raw = self.repo.git.diff("--name-status", "-M50%", old_hash, new_hash)
        result: list[tuple[str, str, str | None]] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                result.append((parts[0], parts[1], None))
            elif len(parts) == 3:
                result.append((parts[0], parts[1], parts[2]))
        return result