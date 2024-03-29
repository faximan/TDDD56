 (let* ((emacs.d "~/.emacs.d/") (plugins-dir  (concat emacs.d "plugins/")))
    (load-file (concat plugins-dir "cua-emul.el"))
    (load-file (concat plugins-dir "encrypt.el"))
    (load-file (concat plugins-dir "flymake.el"))
    (load-file (concat plugins-dir "paredit.el"))
    (load-file (concat plugins-dir "kill-ring-ido.el"))
    (load-file (concat plugins-dir "emacs-for-python/epy-init.el"))

    (setq load-path 
          (append `(,emacs.d ,plugins-dir
                             ,(concat plugins-dir "color-theme")
                             ,(concat plugins-dir "nxml-mode")
                             ;,(concat plugins-dir "slime")
                             ,(concat plugins-dir "clojure-mode")
                             ,(concat plugins-dir "emacs-for-python")
                             ,(concat plugins-dir "ecb")
                                 ,(concat plugins-dir "find-file-in-project")
                                 ,(concat plugins-dir "fuzzy-match")
                                 ,(concat plugins-dir "smex")
                                 ,(concat plugins-dir "js2-mode")
                                 ,(concat plugins-dir "js2-highlight-vars")
                                 ,(concat plugins-dir "expand-region")
                                 ,(concat plugins-dir "mark-multiple")
                                 ,(concat plugins-dir "js2-refactor")
                                 ,(concat plugins-dir "pretty-lambdada")
                                 ,(concat plugins-dir "magit")
                                 ,(concat plugins-dir "wrap-region")
                                 ,(concat plugins-dir "scala-mode2")
                                 ,(concat plugins-dir "ensime/elisp")
                                 ,(concat plugins-dir "toggle-test")
                                  ) load-path)))
